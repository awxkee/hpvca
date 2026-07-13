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

//! A small persistent worker pool with a scoped, borrow-friendly API.
//!
//! Encoding a picture fans out into many short-lived tasks — grid tiles, HEVC
//! tiles, and WPP CTU rows. One pool is created per encode (context-local, not a
//! process-wide global) and feeds all of that encode's parallel regions from a
//! single queue, so whenever a wavefront stalls its worker simply picks up the
//! next ready task from any region (e.g. another tile's WPP row) — cores stay
//! saturated. The pool's threads are joined when it is dropped at the end of the
//! encode, so no worker threads outlive the work they serve.
//!
//! [`ThreadPool::scoped`] mirrors [`std::thread::scope`]: spawned closures may
//! borrow the caller's stack, and the call blocks (the caller helping to run
//! tasks) until every task of that scope has finished. That completion barrier is
//! what makes the internal lifetime erasure sound.

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
        self.shared.shutdown.store(true, Ordering::Release);
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
    _marker: PhantomData<&'scope ()>,
}

impl<'scope> Scope<'scope> {
    /// Queue `task` to run on the pool. It may borrow data living at least for
    /// `'scope`
    pub(crate) fn spawn<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        self.pending.fetch_add(1, Ordering::SeqCst);
        // `pending` lives on the caller's stack for at least `'scope`.
        let pending: &'scope AtomicUsize = unsafe { &*(&self.pending as *const AtomicUsize) };
        let job: Box<dyn FnOnce() + Send + 'scope> = Box::new(move || {
            task();
            pending.fetch_sub(1, Ordering::SeqCst);
        });
        // SAFETY: erase the `'scope` lifetime for the type-erased queue. `wait`
        // blocks until `pending` reaches zero, i.e. until this task has run to
        // completion, so the task never outlives the data it borrows.
        let job: Task =
            unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + 'scope>, Task>(job) };
        self.shared.push(job);
    }

    /// Block until every spawned task has finished, running queued tasks on the
    /// calling thread meanwhile so it contributes instead of idling.
    fn wait(&self) {
        while self.pending.load(Ordering::SeqCst) != 0 {
            match self.shared.try_pop() {
                Some(task) => task(),
                None => std::thread::yield_now(),
            }
        }
    }
}
