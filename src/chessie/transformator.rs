use crate::utl::*;
use anyhow::Result;
use image::{self, imageops, Luma};
use imageproc::geometric_transformations as gt;
use std::f32::consts::PI;

#[allow(unused)]
pub fn manual_transform(img: &LumaImg<u8>) -> Result<()> {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let (px, py) = (100., h * 100. / w);

    /* transformation table
        UL: (0, 0) -> (469, 232)
        UR: (1, 0) -> (910, 253)
        DR: (1, 1) -> (1017, 672)
        DL: (0, 1) -> (363, 642)
    */
    let from = [(469., 232.), (910., 253.), (1017., 672.), (363., 642.)];
    let to = [(px, py), (w - px, py), (w - px, h - py), (px, h - py)];

    let proj = gt::Projection::from_control_points(from, to).unwrap();
    let warped = gt::warp(img, &proj, gt::Interpolation::Nearest, Luma([255]));

    let resized = imageops::resize(
        &warped,
        img.width(),
        img.width(),
        imageops::FilterType::Nearest,
    );
    let rotated =
        gt::rotate_about_center(&resized, PI / 2., gt::Interpolation::Nearest, Luma([255]));

    // imshow!(&out);
    rotated.save("assets/results/game.jpg")?;
    Ok(())
}
