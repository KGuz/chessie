use super::{ndarray_ext::IntoArray, traits::Number};
use image::{ImageBuffer, Luma, Pixel, Primitive, Rgb};
use imageproc::{definitions::Clamp, filter};
use ndarray::{Array, Ix2, Ix3};

pub type RgbImg<A> = ImageBuffer<Rgb<A>, Vec<A>>;
pub type LumaImg<A> = ImageBuffer<Luma<A>, Vec<A>>;

pub trait IntoImage<P: Pixel> {
    fn into_image(self) -> ImageBuffer<P, Vec<P::Subpixel>>;
}
pub trait AsImage<P: Pixel> {
    fn as_image(&self) -> ImageBuffer<P, Vec<P::Subpixel>>;
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

pub struct Gradients<T: Number> {
    pub x: ndarray::Array2<T>,
    pub y: ndarray::Array2<T>,
    pub xx: ndarray::Array2<T>,
    pub yy: ndarray::Array2<T>,
    pub xy: ndarray::Array2<T>,
}

impl<T: Number + Clamp<T>> Gradients<T> {
    pub fn compute(mat: &LumaImg<T>) -> Gradients<T> {
        let (h, v) = (Self::horizontal_sobel(), Self::vertical_sobel());
        let xmat = filter::filter3x3(mat, &h);
        let ymat = filter::filter3x3(mat, &v);

        let xx = filter::filter3x3(&xmat, &h).into_array();
        let yy = filter::filter3x3(&ymat, &v).into_array();
        let xy = filter::filter3x3(&xmat, &v).into_array();
        let y = ymat.into_array();
        let x = xmat.into_array();
        Self { x, y, xx, yy, xy }
    }

    fn vertical_sobel<V: Number>() -> [V; 9] {
        let (n1, n2) = (
            V::from(-1).unwrap_or_default(),
            V::from(-2).unwrap_or_default(),
        );
        let (p1, p2) = (
            V::from(1).unwrap_or_default(),
            V::from(2).unwrap_or_default(),
        );
        let z0 = V::from(0.).unwrap_or_default();
        [n1, n2, n1, z0, z0, z0, p1, p2, p1]
    }

    fn horizontal_sobel<V: Number>() -> [V; 9] {
        let (n1, n2) = (
            V::from(-1).unwrap_or_default(),
            V::from(-2).unwrap_or_default(),
        );
        let (p1, p2) = (
            V::from(1).unwrap_or_default(),
            V::from(2).unwrap_or_default(),
        );
        let z0 = V::from(0.).unwrap_or_default();
        [n1, z0, p1, n2, z0, p2, n1, z0, p1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2, Array3};

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
}
