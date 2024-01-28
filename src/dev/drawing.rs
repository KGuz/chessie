use image::{EncodableLayout, ImageBuffer, PixelWithColorType};
use std::{io, path::Path, process};

pub fn display<P, Container>(image: &ImageBuffer<P, Container>)
where
    P: PixelWithColorType,
    [P::Subpixel]: EncodableLayout,
    Container: std::ops::Deref<Target = [P::Subpixel]>,
{
    let mut path = std::env::temp_dir();
    path.push(format!("{}.jpg", uuid::Uuid::new_v4()));

    match image.save(&path) {
        Err(err) => eprintln!("Err: Failed to display an image: {err}"),
        _ => open_with_cmd(path.to_str().expect("path is valid utf8")),
    }
}

fn open_with_cmd(filepath: &str) {
    process::Command::new("cmd")
        .arg("/C")
        .arg(format!("start {filepath}"))
        .output()
        .map_err(|err| eprintln!("Err: Failed to execute process: {err}"));
}
