use super::*;
use image::GrayImage;
use nalgebra::DMatrix;

pub fn to_mat(image: &GrayImage) -> DMatrix<f32> {
    let (height, width) = (image.height() as usize, image.width() as usize);
    DMatrix::from_row_iterator(height, width, image.iter().map(|v| *v as f32 / 255.))
}

pub fn to_img(matrix: &DMatrix<f32>) -> GrayImage {
    let (nrows, ncols) = matrix.shape();
    GrayImage::from_fn(ncols as u32, nrows as u32, |x, y| {
        let val = matrix[(y as usize, x as usize)];
        image::Luma([(val * 255.) as u8])
    })
}
