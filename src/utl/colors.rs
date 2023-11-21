#![allow(unused)]
use lazy_static::lazy_static;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

#[rustfmt::skip]
const COLORS: [[u8; 3]; 7] = [
    [249,  65,  68],
    [243, 114,  44],
    [248, 150,  30],
    [249, 199,  79],
    [144, 190, 109],
    [ 67, 170, 139],
    [ 87, 117, 144],
];

lazy_static! {
    static ref COLOR_IDX: AtomicUsize = AtomicUsize::new(0);
}

pub fn next() -> [u8; 3] {
    let idx = COLOR_IDX.load(SeqCst);
    COLOR_IDX.store((idx + 1) % COLORS.len(), SeqCst);
    COLORS[idx]
}
