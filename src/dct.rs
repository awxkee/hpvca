//! HEVC coefficient scan orders.
//!
//! The integer transform and quantisation live in `hevc_transform`; this module
//! only provides the diagonal scan tables that map a transform block to the
//! coefficient order used by `residual_coding` (HEVC §6.5.3). `(row, col)` pairs.

/// HEVC up-right diagonal scan for an 8×8 block, `(row, col)`.
#[rustfmt::skip]
pub static ZIGZAG: [(usize, usize); 64] = [
    (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),
    (1,2),(0,3),(3,1),(2,2),(1,3),(3,2),(2,3),(3,3),
    (4,0),(5,0),(4,1),(6,0),(5,1),(4,2),(7,0),(6,1),
    (5,2),(4,3),(7,1),(6,2),(5,3),(7,2),(6,3),(7,3),
    (0,4),(1,4),(0,5),(2,4),(1,5),(0,6),(3,4),(2,5),
    (1,6),(0,7),(3,5),(2,6),(1,7),(3,6),(2,7),(3,7),
    (4,4),(5,4),(4,5),(6,4),(5,5),(4,6),(7,4),(6,5),
    (5,6),(4,7),(7,5),(6,6),(5,7),(7,6),(6,7),(7,7),
];

/// HEVC up-right diagonal scan for a single 4×4 block, `(row, col)`.
#[rustfmt::skip]
pub static DIAG_SCAN_4X4: [(usize, usize); 16] = [
    (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),(2,1),
    (1,2),(0,3),(3,1),(2,2),(1,3),(3,2),(2,3),(3,3),
];
