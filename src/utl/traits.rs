use image::Primitive;
use num::Num;
use serde::Serialize;
use std::fmt::Debug;

pub trait Number: Primitive + Serialize + Default + Num + Debug {}

macro_rules! impl_number {
    ($($T: ty),*) => {$(
        impl Number for $T {}
    )*};
}
impl_number!(usize, u8, u16, u32, u64, isize, i8, i16, i32, i64, f32, f64);
