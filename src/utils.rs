#![allow(dead_code, unused)]
use crate::conversions::*;
use image::{ImageBuffer, Luma, Primitive};
use imageproc::{
    contours::Contour, definitions::Clamp, geometric_transformations,
    geometric_transformations::Projection, point::Point,
};
use itertools::Itertools;
use ndarray::prelude::*;
use plotly::{
    common::{ColorScale, ColorScalePalette},
    layout::{LayoutTemplate, Template},
    HeatMap, Layout, Plot, Trace,
};
use serde::Serialize;

pub trait Number: Primitive + Serialize + Default + num::Num {}
macro_rules! impl_number {
    ($($T: ty),*) => {$(
        impl Number for $T {}
    )*};
}
impl_number!(usize, u8, u16, u32, u64, isize, i8, i16, i32, i64, f32, f64);

/////////////////////////////////// Plotting //////////////////////////////////

type BoxHeatMap<T> = Box<HeatMap<f64, f64, Vec<T>>>;
pub trait AsHeatmap<T: Number> {
    fn as_heatmap(&self) -> BoxHeatMap<T>;
    fn dimensions(&self) -> (usize, usize);
}

impl<T: Number> AsHeatmap<T> for Vec<Vec<T>> {
    fn as_heatmap(&self) -> BoxHeatMap<T> {
        HeatMap::new_z(self.clone())
    }
    fn dimensions(&self) -> (usize, usize) {
        (self[0].len(), self.len())
    }
}

impl<T: Number, const N: usize, const M: usize> AsHeatmap<T> for [[T; M]; N] {
    fn as_heatmap(&self) -> BoxHeatMap<T> {
        let z = self.iter().map(|row| row.to_vec()).collect::<Vec<_>>();
        HeatMap::new_z(z)
    }
    fn dimensions(&self) -> (usize, usize) {
        (self[0].len(), self.len())
    }
}

impl<T: Number> AsHeatmap<T> for LumaImg<T> {
    fn as_heatmap(&self) -> BoxHeatMap<T> {
        let (height, width) = (self.height() as usize, self.width() as usize);
        let mut z = vec![vec![T::default(); width]; height];
        for (x, y, px) in self.enumerate_pixels() {
            let (x, y, v) = (x as usize, y as usize, px.0[0]);
            z[y][x] = v;
        }
        HeatMap::new_z(z)
    }

    fn dimensions(&self) -> (usize, usize) {
        let (w, h) = self.dimensions();
        (w as usize, h as usize)
    }
}

impl<T: Number> AsHeatmap<T> for Array2<T> {
    fn as_heatmap(&self) -> BoxHeatMap<T> {
        let (height, width) = self.dim();
        let mut z = vec![vec![T::default(); width]; height];
        for ((y, x), &v) in self.indexed_iter() {
            z[y][x] = v;
        }
        HeatMap::new_z(z)
    }

    fn dimensions(&self) -> (usize, usize) {
        let (h, w) = self.dim();
        (w, h)
    }
}

macro_rules! imshow {
    ($image: expr, $colormap: expr) => {{
        use plotly::{
            common::{AxisSide, ColorScale},
            layout::Axis,
            Layout, Plot,
        };

        let heatmap = $image
            .as_heatmap()
            .color_scale(ColorScale::Palette($colormap));
        let (width, height) = $image.dimensions();

        let x_axis = Axis::default().range(vec![0, width]).side(AxisSide::Top);
        let y_axis = Axis::default().range(vec![height, 0]);
        let layout = Layout::new()
            .width(2560 - 128)
            .height(1440 - 72)
            .x_axis(x_axis)
            .y_axis(y_axis);

        let mut plot = Plot::new();
        plot.add_trace(heatmap.clone());
        plot.set_layout(layout);
        show_plot(&plot);
    }};
    ($image: expr) => {
        imshow!($image, plotly::common::ColorScalePalette::Greys)
    };
}
pub(crate) use imshow;

macro_rules! trace {
    ($y: expr) => {
        plotly::Scatter::<u32, _>::default().y($y.to_vec())
    };

    ($x: expr, $y: expr) => {
        plotly::Scatter::new($x.to_vec(), $y.to_vec())
    };

    ($x: expr, $y: expr, $style: expr) => {
        plotly::Scatter::new($x.to_vec(), $y.to_vec())
    };
}
pub(crate) use trace;

macro_rules! plot {
    ($($trace: expr),*) => {{
        let mut plot = plotly::Plot::new();
        $(
            plot.add_trace($trace);
        )*
        show_plot(&plot);
    }};
}
pub(crate) use plot;

/// Taken from plotly implementation but here works...
pub fn show_plot(plot: &Plot) {
    open_with_cmd(&save_plot(plot));
}

fn save_plot(plot: &Plot) -> String {
    use rand::distributions::{Alphanumeric, DistString};
    use std::{env, fs::File, io::Write};

    let rendered = plot.to_html();

    // Set up the temp file with a unique filename.
    let mut temp = env::temp_dir();
    let mut plot_name = Alphanumeric.sample_string(&mut rand::thread_rng(), 22);
    plot_name.push_str(".html");
    plot_name = format!("plotly_{}", plot_name);
    temp.push(plot_name);

    // Save the rendered plot to the temp file.
    let temp_path = temp.to_str().unwrap();
    let mut file = File::create(temp_path).unwrap();
    file.write_all(rendered.as_bytes())
        .expect("failed to write html output");
    file.flush().unwrap();

    temp_path.to_string()
}

fn open_with_cmd(temp_path: &str) {
    // Hand off the job of opening the browser to an OS-specific implementation.
    std::process::Command::new("cmd")
        .arg("/C")
        .arg(format!(r#"start {}"#, temp_path))
        .output()
        .expect("Default html app not found");
}

///////////////////////////////// Ndarray Misc ////////////////////////////////

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

#[allow(unused)]
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
////////////////////////////////// Image Misc /////////////////////////////////

pub struct Gradients<T: Number> {
    pub x: Array2<T>,
    pub y: Array2<T>,
    pub xx: Array2<T>,
    pub yy: Array2<T>,
    pub xy: Array2<T>,
}

impl<T: Number + Clamp<T>> Gradients<T> {
    pub fn compute(mat: &LumaImg<T>) -> Gradients<T> {
        let [h, v] = Self::get_sobel_kernels();
        let xmat = imageproc::filter::filter3x3(mat, &h);
        let ymat = imageproc::filter::filter3x3(mat, &v);

        let xx = imageproc::filter::filter3x3(&xmat, &h).into_array();
        let yy = imageproc::filter::filter3x3(&ymat, &v).into_array();
        let xy = imageproc::filter::filter3x3(&xmat, &v).into_array();
        let y = ymat.into_array();
        let x = xmat.into_array();
        Self { x, y, xx, yy, xy }
    }

    fn get_sobel_kernels() -> [[T; 9]; 2] {
        let (n1, n2) = (
            T::from(-1).unwrap_or_default(),
            T::from(-2).unwrap_or_default(),
        );
        let (p1, p2) = (
            T::from(1).unwrap_or_default(),
            T::from(2).unwrap_or_default(),
        );
        let z0 = T::from(0.).unwrap_or_default();

        let h = [n1, z0, p1, n2, z0, p2, n1, z0, p1];
        let v = [n1, n2, n1, z0, z0, z0, p1, p2, p1];
        [h, v]
    }
}

pub struct Visualize;
impl Visualize {
    const SCREEN_RES: (usize, usize) = (2560, 1440);
    const IMAGE_RES: (usize, usize) = (1280, 720);

    fn basic_layout() -> plotly::Layout {
        plotly::Layout::new()
            .width(Self::SCREEN_RES.0 - 128)
            .height(Self::SCREEN_RES.1 - 72)
    }

    fn image_layout() -> plotly::Layout {
        use plotly::{common::AxisSide, layout::Axis};
        let x_axis = Axis::default()
            .range(vec![0, Self::IMAGE_RES.0])
            .side(AxisSide::Top);
        let y_axis = Axis::default().range(vec![Self::IMAGE_RES.1, 0]);
        Self::basic_layout().x_axis(x_axis).y_axis(y_axis)
    }

    pub fn contours<T: Number + 'static>(contours: &[Contour<T>]) {
        let mut plot = Plot::new();

        for c in contours {
            let (x, y) = c.points.iter().cloned().map(|p| (p.x, p.y)).unzip();
            plot.add_trace(plotly::Scatter::new(x, y))
        }

        plot.set_layout(Self::image_layout());
        show_plot(&plot);
    }

    pub fn points<T: Number + 'static>(points: &[Point<T>]) {
        let mut plot = Plot::new();

        let (x, y) = points.iter().cloned().map(|p| (p.x, p.y)).unzip();
        let trace =
            plotly::Scatter::new(x, y).mode(plotly::common::Mode::Markers) as Box<dyn Trace>;
        plot.add_trace(trace);
        plot.set_layout(Self::image_layout());
        show_plot(&plot);
    }

    pub fn grid<T: Number + 'static>(grid: &Array2<T>) {
        assert_eq!(grid.len_of(Axis(1)), 2);
        let mut plot = plotly::Plot::new();

        let (x, y): (Vec<_>, Vec<_>) = grid.axis_iter(Axis(0)).map(|a| (a[1], a[0])).unzip();
        plot.add_trace(plotly::Scatter::new(x, y).mode(plotly::common::Mode::Markers));
        plot.set_layout(Self::basic_layout());
        show_plot(&plot);
    }
}
