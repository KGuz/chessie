use image::{ImageBuffer, Luma, Pixel, Primitive, Rgb};
use imageproc::{geometric_transformations::Projection, point::Point};
use ndarray::prelude::*;

pub type RgbImg<A> = ImageBuffer<Rgb<A>, Vec<A>>;
pub type LumaImg<A> = ImageBuffer<Luma<A>, Vec<A>>;

////////////////////////////// Trait Definitions //////////////////////////////

pub trait IntoArray<A, D> {
    fn into_array(self) -> Array<A, D>;
}
pub trait AsArray<A, D> {
    fn as_array(&self) -> Array<A, D>;
}

pub trait IntoImage<P: Pixel> {
    fn into_image(self) -> ImageBuffer<P, Vec<P::Subpixel>>;
}
pub trait AsImage<P: Pixel> {
    fn as_image(&self) -> ImageBuffer<P, Vec<P::Subpixel>>;
}

pub trait IntoPair<A> {
    fn into_pair(self) -> (A, A);
}
pub trait AsPair<A> {
    fn as_pair(&self) -> (A, A);
}

pub trait IntoPoint<A> {
    fn into_point(self) -> Point<A>;
}
pub trait AsPoint<A> {
    fn as_point(&self) -> Point<A>;
}
/////////////////////////////// Implementations ///////////////////////////////

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

impl<A: Primitive> IntoImage<Luma<A>> for Array<A, Ix2> {
    fn into_image(self) -> LumaImg<A> {
        assert!(self.is_standard_layout());

        let (height, width) = self.dim();
        LumaImg::from_raw(width as u32, height as u32, self.into_raw_vec())
            .expect("Container should have the right size for the image dimensions")
    }
}

macro_rules! impl_into_image {
    ($($A: ty),*) => {$(
        impl IntoImage<Rgb<$A>> for Array<$A, Ix3> {
            fn into_image(self) -> RgbImg<$A> {
                assert!(self.is_standard_layout());

                let (height, width, _) = self.dim();
                RgbImg::from_raw(width as u32, height as u32, self.into_raw_vec())
                    .expect("Container should have the right size for the image dimensions")
            }
        }
    )*};
} // As long as Enlargeable trait is not exposed from image crate, conversions must be implemented per type
impl_into_image!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl<A: Primitive> AsImage<Luma<A>> for Array<A, Ix2> {
    fn as_image(&self) -> LumaImg<A> {
        assert!(self.is_standard_layout());

        let (height, width) = self.dim();
        LumaImg::from_raw(width as u32, height as u32, self.clone().into_raw_vec())
            .expect("Container should have the right size for the image dimensions")
    }
}

macro_rules! impl_as_image {
    ($($A: ty),*) => {$(
        impl AsImage<Rgb<$A>> for Array<$A, Ix3> {
            fn as_image(&self) -> RgbImg<$A> {
                assert!(self.is_standard_layout());

                let (height, width, _) = self.dim();
                RgbImg::from_raw(width as u32, height as u32, self.clone().into_raw_vec())
                    .expect("Container should have the right size for the image dimensions")
            }
        }
    )*};
} // As long as Enlargeable trait is not exposed from image crate, conversions must be implemented per type
impl_as_image!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl<A> IntoPair<A> for Point<A> {
    fn into_pair(self) -> (A, A) {
        (self.x, self.y)
    }
}

impl<A> IntoPair<A> for [A; 2] {
    fn into_pair(self) -> (A, A) {
        let [x, y] = self;
        (x, y)
    }
}

impl<A: Clone> AsPair<A> for Point<A> {
    fn as_pair(&self) -> (A, A) {
        (self.x.clone(), self.y.clone())
    }
}

impl<A: Clone> AsPair<A> for [A; 2] {
    fn as_pair(&self) -> (A, A) {
        (self[0].clone(), self[1].clone())
    }
}

impl<A: Clone> AsPair<A> for Vec<A> {
    fn as_pair(&self) -> (A, A) {
        assert!(self.len() == 2);
        (self[0].clone(), self[1].clone())
    }
}

impl<A: Clone> AsPair<A> for &[A] {
    fn as_pair(&self) -> (A, A) {
        assert!(self.len() == 2);
        (self[0].clone(), self[1].clone())
    }
}

impl<A> IntoPoint<A> for [A; 2] {
    fn into_point(self) -> Point<A> {
        let [x, y] = self;
        Point { x, y }
    }
}

impl<A: Copy> AsPoint<A> for [A; 2] {
    fn as_point(&self) -> Point<A> {
        let &[x, y] = self;
        Point { x, y }
    }
}

impl<A> IntoPoint<A> for (A, A) {
    fn into_point(self) -> Point<A> {
        let (x, y) = self;
        Point { x, y }
    }
}

impl<A: Copy> AsPoint<A> for (A, A) {
    fn as_point(&self) -> Point<A> {
        let &(x, y) = self;
        Point { x, y }
    }
}

////////////////////////////// Conversions Tests //////////////////////////////

#[cfg(test)]
mod conversions_tests {
    use super::*;

    #[test]
    fn array3_into_rgbimg() {
        let arr: Array3<u8> = array![
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
            [[7, 7, 7], [8, 8, 8], [9, 9, 9]],
        ];
        let array_ptr = arr.as_ptr();
        let array_img = arr.into_image();

        assert_eq!(array_ptr, array_img.as_ptr());
        assert_eq!(array_img.get_pixel(0, 0), &Rgb([1, 1, 1]));
    }

    #[test]
    fn array2_into_lumaimg() {
        let arr: Array2<i16> = array![[1, 2, 3], [4, 5, 6], [7, 8, 9],];
        let array_ptr = arr.as_ptr();
        let array_img = arr.into_image();

        assert_eq!(array_ptr, array_img.as_ptr());
        assert_eq!(array_img.get_pixel(1, 1), &Luma([5]));
    }

    #[test]
    fn array3_as_rgbimg() {
        let arr: Array3<u32> = array![
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
            [[7, 7, 7], [8, 8, 8], [9, 9, 9]],
        ];
        let array_ptr = arr.as_ptr();
        let array_img = arr.as_image();

        assert_ne!(array_ptr, array_img.as_ptr());
        assert_eq!(array_img.get_pixel(2, 2), &Rgb([9, 9, 9]));
    }

    #[test]
    fn array2_as_lumaimg() {
        let arr: Array2<f64> = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.],];
        let array_ptr = arr.as_ptr();
        let array_img = arr.as_image();

        assert_ne!(array_ptr, array_img.as_ptr());
        assert_eq!(array_img.get_pixel(0, 1), &Luma([4.]));
    }

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
        let img: RgbImg<u16> = RgbImg::from_vec(
            3,
            3,
            vec![
                1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
            ],
        )
        .unwrap();
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
        let img: RgbImg<f32> = RgbImg::from_vec(
            3,
            3,
            vec![
                1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4., 5., 5., 5., 6., 6., 6., 7., 7., 7.,
                8., 8., 8., 9., 9., 9.,
            ],
        )
        .unwrap();
        let image_ptr = img.as_ptr();
        let image_arr = img.as_array();

        assert_ne!(image_ptr, image_arr.as_ptr());
        assert_eq!(image_arr[[0, 2, 0]], 3.);
    }

    #[test]
    fn point_into_pair() {
        assert_eq!((0, 0), Point { x: 0, y: 0 }.into_pair());
    }

    #[test]
    fn arr_into_pair() {
        assert_eq!((5., 6.), [5., 6.].into_pair());
    }

    #[test]
    fn point_as_pair() {
        assert_eq!((1, 2), Point { x: 1, y: 2 }.as_pair());
    }

    #[test]
    fn arr_as_pair() {
        assert_eq!((5., 6.), [5., 6.].as_pair());
    }

    #[test]
    fn vec_as_pair() {
        assert_eq!((-3, -4), vec![-3, -4].as_pair());
    }

    #[test]
    fn slice_as_pair() {
        assert_eq!((-7, -8), [-7, -8].as_slice().as_pair());
    }
}
