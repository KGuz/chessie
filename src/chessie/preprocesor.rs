use image::{self, imageops};

pub fn preprocess(src: &image::RgbImage) -> image::GrayImage {
    let img = imageops::resize(src, 1280, 720, imageops::FilterType::Nearest);
    let img = imageops::grayscale(&img);
    imageproc::filter::median_filter(&img, 1, 1)
}
