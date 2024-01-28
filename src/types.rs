use image::{ImageBuffer, Luma, Rgb};
use nalgebra::Point2;

pub type Result<A> = core::result::Result<A, &'static str>;
pub type Quad<A> = [Point2<A>; 4];
pub type RgbImg<A> = ImageBuffer<Rgb<A>, Vec<A>>;
pub type LumaImg<A> = ImageBuffer<Luma<A>, Vec<A>>;
