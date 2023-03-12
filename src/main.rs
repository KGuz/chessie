use anyhow::Result;
use chessie::{detector, parser};
use image::{self, imageops};
use imageproc::filter;

fn main() -> Result<()> {
    let args = parser::args();
    let src = image::open(args.file)?.to_rgb8();

    let img = preprocess(&src);
    detector::detect_board(&img)?;
    Ok(())
}

fn preprocess(src: &image::RgbImage) -> image::GrayImage {
    let img = imageops::resize(src, 1280, 720, imageops::FilterType::Nearest);
    let img = imageops::grayscale(&img);
    filter::median_filter(&img, 1, 1)
}
