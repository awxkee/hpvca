/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// A queued unit of work. Erased to `'static`; every task is guaranteed by the
/// owning [`Scope`] to complete before the borrowed data it captured is freed.
type Task = Box<dyn FnOnce() + Send>;

struct Shared {
    queue: Mutex<VecDeque<Task>>,
    ready: Condvar,
    /// Set by [`ThreadPool::drop`] to wake and retire idle workers.
    shutdown: AtomicBool,
}

impl Shared {
    fn push(&self, task: Task) {
        self.queue.lock().unwrap().push_back(task);
        self.ready.notify_one();
    }
    /// Pop one task without blocking.
    fn try_pop(&self) -> Option<Task> {
        self.queue.lock().unwrap().pop_front()
    }
}

/// A pool of worker threads owned by whoever created it — created per parallel
/// encode region rather than living in a process-wide global, so its lifetime is
/// scoped to the work it serves and its threads are joined when it is dropped.
pub(crate) struct ThreadPool {
    shared: Arc<Shared>,
    workers: usize,
    handles: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Create a pool with `workers` background threads. The creating thread also
    /// runs tasks inside [`scoped`](Self::scoped), so total concurrency is
    /// `workers + 1`; pass `desired_parallelism - 1`.
    pub(crate) fn new(workers: usize) -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            ready: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let shared = shared.clone();
            // Parks on the condvar while idle; retires when `shutdown` is set.
            if let Ok(h) = std::thread::Builder::new()
                .name("hpvca-pool".into())
                .spawn(move || worker_loop(&shared))
            {
                handles.push(h);
            }
        }
        let workers = handles.len();
        ThreadPool {
            shared,
            workers,
            handles,
        }
    }

    /// Total concurrency available inside a scope: the workers plus the calling
    /// thread, which helps run tasks while it waits.
    pub(crate) fn parallelism(&self) -> usize {
        self.workers + 1
    }

    /// Run `f`, which may [`spawn`](Scope::spawn) tasks onto the pool. Blocks
    /// until every task spawned in the scope has completed; the calling thread
    /// helps drain the queue meanwhile, so it is never merely idle.
    pub(crate) fn scoped<'scope, F, R>(&'scope self, f: F) -> R
    where
        F: FnOnce(&Scope<'scope>) -> R,
    {
        let scope = Scope {
            shared: &self.shared,
            pending: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            _marker: PhantomData,
        };
        let out = f(&scope);
        scope.wait();
        out
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Signal retirement and wake every parked worker, then join. Any tasks
        // still queued at drop are drained by the workers before they see an empty
        // queue + shutdown; in practice `scoped` has already awaited all of them.
        //
        // The flag MUST be set while holding the queue mutex. A worker parks with
        // `ready.wait(q)`, which releases that mutex atomically; between its
        // `shutdown` check and that release it still holds the lock. Storing the
        // flag without the lock lets `notify_all` land in exactly that gap, where
        // there is no waiter yet to receive it — the worker then parks forever and
        // `join` below never returns. Taking the lock orders us either fully
        // before the check (worker sees the flag and exits) or fully after the
        // park (worker receives the notification).
        {
            let _guard = self.shared.queue.lock().unwrap();
            self.shared.shutdown.store(true, Ordering::Release);
        }
        self.shared.ready.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker_loop(shared: &Shared) {
    loop {
        let task = {
            let mut q = shared.queue.lock().unwrap();
            loop {
                if let Some(t) = q.pop_front() {
                    break Some(t);
                }
                if shared.shutdown.load(Ordering::Acquire) {
                    break None;
                }
                q = shared.ready.wait(q).unwrap();
            }
        };
        match task {
            Some(t) => t(),
            None => return,
        }
    }
}

/// Handle for spawning borrowed tasks within [`ThreadPool::scoped`].
pub(crate) struct Scope<'scope> {
    shared: &'scope Shared,
    pending: AtomicUsize,
    /// Set when a task unwound. The panic is re-raised on the scope owner in
    /// [`Scope::wait`] so a worker-thread failure surfaces as a normal panic
    /// instead of vanishing with the worker.
    panicked: AtomicBool,
    _marker: PhantomData<&'scope ()>,
}

struct PendingGuard<'a> {
    pending: &'a AtomicUsize,
    panicked: &'a AtomicBool,
}

impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.panicked.store(true, Ordering::SeqCst);
        }
        self.pending.fetch_sub(1, Ordering::SeqCst);
    }
}

impl<'scope> Scope<'scope> {
    /// Queue `task` to run on the pool. It may borrow data living at least for
    /// `'scope`
    pub(crate) fn spawn<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        self.pending.fetch_add(1, Ordering::SeqCst);
        // `pending`/`panicked` live on the caller's stack for at least `'scope`.
        let pending: &'scope AtomicUsize = unsafe { &*(&self.pending as *const AtomicUsize) };
        let panicked: &'scope AtomicBool = unsafe { &*(&self.panicked as *const AtomicBool) };
        let job: Box<dyn FnOnce() + Send + 'scope> = Box::new(move || {
            // The guard decrements on unwind too, so a panicking task can never
            // strand `wait()`.
            let _guard = PendingGuard { pending, panicked };
            task();
        });
        // SAFETY: erase the `'scope` lifetime for the type-erased queue. `wait`
        // blocks until `pending` reaches zero, i.e. until this task has run to
        // completion, so the task never outlives the data it borrows.
        let job: Task =
            unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + 'scope>, Task>(job) };
        self.shared.push(job);
    }

    fn wait(&self) {
        // Consecutive idle spins (no task to steal, work still outstanding)
        // before declaring the scope stuck. Each idle spin yields or sleeps, so
        // this is minutes of no progress, never a false trip on slow encodes.
        const STALL_LIMIT: u64 = 20_000_000;
        let mut idle_spins: u64 = 0;
        while self.pending.load(Ordering::SeqCst) != 0 {
            match self.shared.try_pop() {
                Some(task) => {
                    idle_spins = 0;
                    task();
                }
                None => {
                    idle_spins += 1;
                    assert!(
                        idle_spins < STALL_LIMIT,
                        "hpvca thread pool stalled: {} task(s) still pending with an empty \
                         queue after {} idle spins — a task neither finished nor unwound",
                        self.pending.load(Ordering::SeqCst),
                        idle_spins,
                    );
                    // Spin briefly (the common case is a task finishing within
                    // microseconds), then back off to sleeping so a long task
                    // does not pin this core at 100%.
                    if idle_spins < 1024 {
                        std::hint::spin_loop();
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(std::time::Duration::from_micros(50));
                    }
                }
            }
        }
        assert!(
            !self.panicked.load(Ordering::SeqCst),
            "hpvca thread pool: a worker task panicked (see the panic printed by the \
             worker thread above)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn panicking_task_does_not_hang_the_scope() {
        let pool = ThreadPool::new(2);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.scoped(|scope| {
                for i in 0..2 {
                    scope.spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(30));
                        panic!("worker task {i} fails on purpose");
                    });
                }
                // Let the workers dequeue both tasks before the caller waits.
                std::thread::sleep(std::time::Duration::from_millis(150));
            });
        }));
        assert!(
            result.is_err(),
            "a task that panicked on a worker must surface as a panic, not a hang"
        );
    }

    #[test]
    fn rapid_pool_create_drop_does_not_lose_shutdown_wakeup() {
        for _ in 0..3000 {
            let pool = ThreadPool::new(4);
            drop(pool);
        }
    }

    #[test]
    fn all_tasks_run_to_completion() {
        let pool = ThreadPool::new(3);
        let counter = AtomicUsize::new(0);
        pool.scoped(|scope| {
            for _ in 0..64 {
                scope.spawn(|| {
                    counter.fetch_add(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(counter.load(Ordering::SeqCst), 64);
    }

    #[test]
    fn zero_worker_pool_still_drains() {
        let pool = ThreadPool::new(0);
        let counter = AtomicUsize::new(0);
        pool.scoped(|scope| {
            for _ in 0..16 {
                scope.spawn(|| {
                    counter.fetch_add(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(counter.load(Ordering::SeqCst), 16);
    }
}
