#![allow(dead_code)]

use super::{
    image_ext::{LumaImg, RgbImg},
    traits::Number,
};
use image::Primitive;
use imageproc::geometric_transformations::Projection;
use ndarray::{Array, Array1, Array2, Array3, Axis, Ix2, Ix3};

pub trait IntoArray<A, D> {
    fn into_array(self) -> Array<A, D>;
}
pub trait AsArray<A, D> {
    fn as_array(&self) -> Array<A, D>;
}

impl<A: Primitive> IntoArray<A, Ix2> for LumaImg<A> {
    fn into_array(self) -> Array<A, Ix2> {
        let shape = (self.height() as usize, self.width() as usize);
        Array2::from_shape_vec(shape, self.into_raw()).unwrap()
    }
}

macro_rules! impl_into_array {
    ($($A: ty),*) => {$(
        impl IntoArray<$A, Ix3> for RgbImg<$A> {
            fn into_array(self) -> Array<$A, Ix3> {
                let shape = (self.height() as usize, self.width() as usize, 3);
                Array3::from_shape_vec(shape, self.into_raw()).unwrap()
            }
        }
    )*};
} // As long as Enlargeable trait is not exposed from image crate, conversions must be implemented per type
impl_into_array!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl<A: Primitive> AsArray<A, Ix2> for LumaImg<A> {
    fn as_array(&self) -> Array<A, Ix2> {
        let shape = (self.height() as usize, self.width() as usize);
        Array2::from_shape_vec(shape, self.to_vec()).unwrap()
    }
}

macro_rules! impl_as_array {
    ($($A: ty),*) => {$(
        impl AsArray<$A, Ix3> for RgbImg<$A> {
            fn as_array(&self) -> Array<$A, Ix3> {
                let shape = (self.height() as usize, self.width() as usize, 3);
                Array3::from_shape_vec(shape, self.to_vec()).unwrap()
            }
        }
    )*};
} // As long as Enlargeable trait is not exposed from image crate, conversions must be implemented per type
impl_as_array!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl AsArray<f32, Ix2> for Projection {
    /// Get transformation matrix by parsing debug information.
    /// Highly inefficient, but imageproc does not implement any type of interface to extract this information
    fn as_array(&self) -> Array<f32, Ix2> {
        let debug_info = format!("{:?}", self);
        let transform: String = debug_info
            .chars()
            .skip_while(|c| c != &'[')
            .skip(1)
            .take_while(|c| c != &']')
            .collect();

        let values = transform.split(", ").map(|num| num.parse::<f32>().unwrap());
        Array::from_iter(values).into_shape((3, 3)).unwrap()
    }
}

macro_rules! amap {
    (@zeros_like $first:expr) => {
        ndarray::Array::zeros($first.raw_dim())
    };
    (@zeros_like $first:expr, $($rest:tt)*) => {
        ndarray::Array::zeros($first.raw_dim())
    };

    (($($val:pat in $arr:expr),+) $body: expr) => {{
        let mut _arr = (amap!(@zeros_like $($arr),*));
        ndarray::azip!((_a in &mut _arr, $($val in $arr),*) *_a = $body);
        _arr
    }};
    (($($val:pat),+) in ($($arr:expr),+) $body: expr) => {{
        let mut _arr = (amap!(@zeros_like $($arr),*));
        ndarray::azip!((_a in &mut _arr, $($val in $arr),*) *_a = $body);
        _arr
    }};
}
pub(crate) use amap;

macro_rules! argwhere {
    ($arr:ident $($rest:tt)*) => {
        $arr.indexed_iter().filter_map(|(idx, &val)| {
            if val $($rest)* { Some(idx) } else { None }
        })
    };
}
pub(crate) use argwhere;

pub trait Misc<T> {
    fn threshold(&mut self, val: T);
    fn count(&self, f: fn(T) -> bool) -> usize;
    fn argwhere(&self, f: fn(T) -> bool) -> Array1<(usize, usize)>;
}

impl<T: Number> Misc<T> for Array2<T> {
    fn threshold(&mut self, val: T) {
        self.mapv_inplace(|a| if a < val { T::default() } else { a });
    }
    fn count(&self, f: fn(T) -> bool) -> usize {
        self.iter().copied().filter(|x| f(*x)).count()
    }
    fn argwhere(&self, f: fn(T) -> bool) -> Array1<(usize, usize)> {
        let f = |(idx, &val)| if f(val) { Some(idx) } else { None };
        Array1::from_iter(self.indexed_iter().filter_map(f))
    }
}

pub fn meshgrid<T: Number>(x: &Array1<T>, y: &Array1<T>) -> Array2<T> {
    let (xlen, ylen) = (x.len(), y.len());
    let mut z = Array::zeros((xlen * ylen, 2));

    for i in 0..z.len_of(Axis(0)) {
        z[[i, 1]] = y[i / xlen];
        z[[i, 0]] = x[i % ylen];
    }
    z
}

fn interlope(n: usize) -> Vec<usize> {
    let mut cycle = true;
    let mut x = (0..n).cycle();
    let mut y = (0..n).cycle();
    (0..n * n)
        .map(|_| {
            cycle = !cycle;
            if cycle {
                x.next().unwrap()
            } else {
                y.next().unwrap()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lumaimg_into_array2() {
        let img: LumaImg<i8> = LumaImg::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let image_ptr = img.as_ptr();
        let image_arr = img.into_array();

        assert_eq!(image_ptr, image_arr.as_ptr());
        assert_eq!(image_arr[[2, 0]], 7);
    }

    #[test]
    fn rgbimg_into_array3() {
        #[rustfmt::skip]
        let img: RgbImg<u16> = RgbImg::from_vec(
            3,
            3,
            vec![
                1, 1, 1,  2, 2, 2,  3, 3, 3, 
                4, 4, 4,  5, 5, 5,  6, 6, 6, 
                7, 7, 7,  8, 8, 8,  9, 9, 9,
            ],
        ).unwrap();

        let image_ptr = img.as_ptr();
        let image_arr = img.into_array();

        assert_eq!(image_ptr, image_arr.as_ptr());
        assert_eq!(image_arr[[2, 1, 0]], 8);
    }

    #[test]
    fn lumaimg_as_array2() {
        let img: LumaImg<i32> = LumaImg::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let image_ptr = img.as_ptr();
        let image_arr = img.as_array();

        assert_ne!(image_ptr, image_arr.as_ptr());
        assert_eq!(image_arr[[0, 1]], 2);
    }

    #[test]
    fn rgbimg_as_array3() {
        #[rustfmt::skip]
        let img: RgbImg<f32> = RgbImg::from_vec(
            3,
            3,
            vec![
                1., 1., 1.,  2., 2., 2.,  3., 3., 3., 
                4., 4., 4.,  5., 5., 5.,  6., 6., 6., 
                7., 7., 7.,  8., 8., 8.,  9., 9., 9.,
            ],
        ).unwrap();

        let image_ptr = img.as_ptr();
        let image_arr = img.as_array();

        assert_ne!(image_ptr, image_arr.as_ptr());
        assert_eq!(image_arr[[0, 2, 0]], 3.);
    }
}
