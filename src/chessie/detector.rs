use crate::utl::{AsArray, *};
use anyhow::Result;
use image::{self, imageops};
use imageproc::{contours::Contour, geometric_transformations::Projection, point::Point, *};
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use plotly::common::ColorScalePalette;
use std::f32::consts::SQRT_2;

pub fn detect_board(src: &LumaImg<u8>) -> Result<()> {
    let img: LumaImg<f32> = imageops::grayscale_with_type(src);

    // finds all the points that can determine the edges of the chess squares
    let sp = saddle_points(&img);
    let mut nms = nonmax_suppression(&sp, 16);

    // disregards influence of subnormal values
    let average = nms.sum() / nms.count(f32::is_normal) as f32;
    nms.threshold(average);

    // transforms 2d array into 1d array of points
    let spts = nms.argwhere(|x| x > 0.);

    // finds all contours inside the image and filters them using previously detected points
    let mut contours = find_contours(src);
    refine_contours(&mut contours, &spts);

    // displays all the data calculated so far
    imshow!(&nms);
    Visualize::contours(&contours);
    Visualize::points(&spts.iter().map(|sp| Point::new(sp.1, sp.0)).collect_vec());

    // TODO:
    // For each contour, treat their points as a perfect square and calculate
    // the projection matrix. Use this projection matrix to generate a set of
    // points around this contour and see how closely they match those
    // previously detected.
    for cnt in contours {
        let projection_matrix = projection_matrix(cnt).unwrap().as_array();
        for n in 1..8 {
            let (_orthogonal_grid_n, _transformed_grid_n) = chess_grid(&projection_matrix, n);
        }
    }

    Ok(())
}

/// Calculates image gradients and finds points on the surface of the graph of
/// those functions where the derivatives in orthogonal directions are zero.
/// These points are then pruned by selecting only a subset of most prominent
/// points.
fn saddle_points(img: &LumaImg<f32>) -> Array2<f32> {
    let g = Gradients::compute(img);
    let mut saddle = amap!((xx in &g.xx, yy in &g.yy, xy in &g.xy) {
        let s = -(xx * yy - xy.powi(2));
        if s < 0. { 0. } else { s }
    });

    // IDEA: sort the points and choose a fixed number of most prominent
    let mut score = saddle.mapv(|s| u32::from(s > 0.)).sum();
    let mut th = 0.5;
    while score > 10000 {
        th *= 2.;
        saddle.threshold(th);
        score = saddle.mapv(|s| u32::from(s > 0.)).sum();
    }
    saddle
}

/// Eliminates duplicate detections and selects only the most prominent point
/// within a window of a given size.
fn nonmax_suppression<T: std::cmp::PartialOrd + Default + Copy>(
    arr: &Array2<T>,
    win: usize,
) -> Array2<T> {
    let mut nms = Array2::default(arr.raw_dim());
    let (h, w) = arr.dim();

    for (y, x) in argwhere!(arr > T::default()) {
        let [y0, yn] = [y.saturating_sub(win), h.min(y + win + 1)];
        let [x0, xn] = [x.saturating_sub(win), w.min(x + win + 1)];
        let value = arr[[y, x]];

        let view = arr.slice(s![y0..yn, x0..xn]);
        if *view.max().unwrap() == value {
            nms[[y, x]] = value;
        }
    }
    nms
}

/// Detects possible chessboard squares by smoothing out the counturs obtained
/// from the Canny Edge Detection algorithm. The contours are then verified
/// using several heuristics typical of rectangles.
fn find_contours(img: &LumaImg<u8>) -> Vec<Contour<u32>> {
    let edges = edges::canny(img, 20., 250.);
    let edges_grad = morphology::dilate(&edges, distance_transform::Norm::L1, 3);
    let contours = contours::find_contours::<u32>(&edges_grad);

    let simplify_contour = |cnt: Contour<_>| -> Contour<_> {
        let arc_len = geometry::arc_length(&cnt.points, true);
        let poly_dp = geometry::approximate_polygon_dp(&cnt.points, 0.05 * arc_len, true);
        Contour::new(poly_dp, cnt.border_type, cnt.parent)
    };

    let is_valid = |cnt: &Contour<_>| -> bool {
        cnt.parent.is_some()
            && cnt.points.len() == 4
            && polygon_area(&cnt.points) > 64.
            && is_square_like(&cnt.points, 0.1)
    };

    contours
        .into_iter()
        .map(simplify_contour)
        .filter(is_valid)
        .collect()
}

/// Compares saddle points to interior points of contours and filters out those
/// contours that do not overlap.
fn refine_contours(contours: &mut Vec<Contour<u32>>, spts: &Array1<(usize, usize)>) {
    let dist = |sp: (usize, usize), cp: &Point<u32>| {
        sp.0.abs_diff(cp.y as usize) + sp.1.abs_diff(cp.x as usize)
    };

    for cnt in contours {
        for cp in &mut cnt.points {
            let argmin = spts.mapv(|sp| dist(sp, cp)).argmin().unwrap();
            let (y, x) = spts[argmin];

            if dist((y, x), cp) < 20 {
                *cp = Point::new(x as u32, y as u32);
            }
        }
    }
}

/// Calculates the area of the surface enclosed by a set of points
fn polygon_area<T: Into<f64> + Copy>(points: &[point::Point<T>]) -> f64 {
    if points.len() < 3 {
        return 0.;
    }

    let mut area = 0.;
    for i in 0..points.len() - 1 {
        let (x1, x2) = (points[i].x.into(), points[i + 1].x.into());
        let (y1, y2) = (points[i].y.into(), points[i + 1].y.into());
        area += (y1 + y2) * (x1 - x2);
    }
    area / 2.
}

/// A heuristic based on the fact that a rectangle should have two pairs of
/// sides of equal lengths and diagonals of equal length.
fn is_square_like<T: Into<f64> + Copy>(points: &[point::Point<T>], epsilon: f64) -> bool {
    let points: Vec<_> = points
        .iter()
        .map(|p| Point::new(p.x.into(), p.y.into()))
        .collect();
    let [a, b, c, d] = points[..] else {
        return false;
    };

    let dist =
        |p1: Point<f64>, p2: Point<f64>| ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt();
    let mut sides = [dist(a, b), dist(b, c), dist(c, d), dist(d, a)];
    let diags = [dist(a, c), dist(b, d)];

    let are_equal = |values: &[f64]| {
        let avg = values.iter().sum::<f64>() / values.len() as f64;
        values
            .iter()
            .map(|x| (x - avg).abs() / avg)
            .all(|e| e < epsilon)
    };

    sides.sort_by(|a, b| a.partial_cmp(b).unwrap());
    are_equal(&sides[..2]) && are_equal(&sides[2..]) && are_equal(&diags)
}

/// Calculates a projection matrix that transforms a given contour into a
/// perfect square. Upper left point is treated as the anchor.
fn projection_matrix(contour: Contour<u32>) -> Option<Projection> {
    let _side_len = |points: [(f32, f32); 4]| {
        let [ul, ur, dr, dl] = points;
        let d1 = ((ul.0 - dr.0).powi(2) + (ul.1 - dr.1).powi(2)).sqrt();
        let d2 = ((ur.0 - dl.0).powi(2) + (ur.1 - dl.1).powi(2)).sqrt();
        0.25 * (d1 + d2) * SQRT_2
    };

    let contour = contour
        .points
        .iter()
        .map(|p| (p.x as f32, p.y as f32))
        .collect_vec();
    let from: [(f32, f32); 4] = contour.as_slice().try_into().ok()?;
    let (x0, y0) = from[0];

    let a = 1.; // side_len(from);
    let to = [(x0, y0), (x0 + a, y0), (x0 + a, y0 + a), (x0, y0 + a)];
    Projection::from_control_points(to, from)
}

/// Generates an array of points in an n by n grid.
/// ```
/// let arr = ndarray::array![
///     [0., 0.], [1., 0.], [2., 0.],
///     [0., 1.], [1., 1.], [2., 1.],
///     [0., 2.], [1., 2.], [2., 2.],
/// ];
/// assert_eq!(chessie::generate_grid(3), arr);
/// ```
pub fn generate_grid(n: usize) -> Array2<f32> {
    let mut z = Array::zeros((n * n, 2));
    for i in 0..z.len_of(Axis(0)) {
        z[[i, 0]] = (i % n) as f32;
        z[[i, 1]] = (i / n) as f32;
    }
    z
}

fn chess_grid(projection_matrix: &Array2<f32>, n: usize) -> (Array2<f32>, Array2<f32>) {
    let len = 2 * (n + 1);
    let orthogonal_grid = generate_grid(len) - n as f32;

    let mut padded_grid = orthogonal_grid.clone();
    padded_grid
        .push_column(Array::ones(len.pow(2)).view())
        .unwrap();

    let mut transformed_grid = padded_grid.dot(&projection_matrix.t());
    transformed_grid
        .axis_iter_mut(Axis(0))
        .for_each(|mut row| row /= row[2]);
    transformed_grid.remove_index(Axis(1), 2);

    (orthogonal_grid, transformed_grid)
}

/* ------------------------------ experiments ------------------------------ */

#[allow(unused)]
fn draw_contours(img: &LumaImg<f32>, contours: &[Contour<u32>]) {
    let mut canvas = img.clone();
    for cnt in contours {
        let poly: Vec<_> = cnt
            .points
            .iter()
            .map(|p| Point::new(p.x as i32, p.y as i32))
            .collect();
        drawing::draw_polygon_mut(&mut canvas, &poly, image::Luma([0.]));
    }
    imshow!(&canvas, ColorScalePalette::Viridis);
}

#[allow(unused)]
fn fit_to_saddle_points(grid: &Array2<f32>, spts: &Array1<(usize, usize)>) {
    dbg!(grid, spts);
    let _n = grid.len_of(Axis(0));
    let _new_grid = grid.clone();

    for point in grid.axis_iter(Axis(0)) {
        let (_x, _y) = (point[0] as i64, point[1] as i64);
    }
}

#[allow(unused)]
fn phase_masked(img: &LumaImg<f32>) -> Array2<f32> {
    let g = Gradients::compute(img);
    let mag = amap!((gx in &g.x, gy in &g.y) f32::sqrt(gx * gx + gy * gy));
    let phase = amap!((x in &g.x, y in &g.y) x.atan2(*y));

    let avg = mag.mean().unwrap();
    amap!((m in &mag, p in &phase) {
        if *m < 2. * avg { f32::NAN } else { *p }
    })
}

#[allow(unused)]
fn detect_lines(img: &LumaImg<u8>) -> Vec<hough::PolarLine> {
    let [th1, th2, votes] = [120, 140, 90];

    let canny = edges::canny(img, th1 as f32, th2 as f32);
    let options = hough::LineDetectionOptions {
        vote_threshold: votes as u32,
        suppression_radius: 2,
    };
    hough::detect_lines(&canny, options)
}

#[allow(unused)]
fn morphology_gradient(image: &LumaImg<u8>, norm: distance_transform::Norm, k: u8) -> LumaImg<u8> {
    let dilate = morphology::dilate(image, norm, k);
    let erode = morphology::erode(image, norm, k);
    (dilate.into_array() - erode.into_array()).into_image()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generate_grid_test() {
        #[rustfmt::skip]
        let expected = array![
            [0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.],
            [0., 1.], [1., 1.], [2., 1.], [3., 1.], [4., 1.],
            [0., 2.], [1., 2.], [2., 2.], [3., 2.], [4., 2.],
            [0., 3.], [1., 3.], [2., 3.], [3., 3.], [4., 3.],
            [0., 4.], [1., 4.], [2., 4.], [3., 4.], [4., 4.],
        ];
        let output = generate_grid(5);
        assert_eq!(output, expected);

        #[rustfmt::skip]
        let expected = array![
            [0., 0.], [1., 0.], [2., 0.],
            [0., 1.], [1., 1.], [2., 1.],
            [0., 2.], [1., 2.], [2., 2.],
        ];
        let output = generate_grid(3);
        assert_eq!(output, expected);

        let expected = array![[0., 0.]];
        let output = generate_grid(1);
        assert_eq!(output, expected);

        #[rustfmt::skip]
        let expected = array![
            [0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.], [6., 0.], [7., 0.],
            [0., 1.], [1., 1.], [2., 1.], [3., 1.], [4., 1.], [5., 1.], [6., 1.], [7., 1.],
            [0., 2.], [1., 2.], [2., 2.], [3., 2.], [4., 2.], [5., 2.], [6., 2.], [7., 2.],
            [0., 3.], [1., 3.], [2., 3.], [3., 3.], [4., 3.], [5., 3.], [6., 3.], [7., 3.],
            [0., 4.], [1., 4.], [2., 4.], [3., 4.], [4., 4.], [5., 4.], [6., 4.], [7., 4.],
            [0., 5.], [1., 5.], [2., 5.], [3., 5.], [4., 5.], [5., 5.], [6., 5.], [7., 5.],
            [0., 6.], [1., 6.], [2., 6.], [3., 6.], [4., 6.], [5., 6.], [6., 6.], [7., 6.],
            [0., 7.], [1., 7.], [2., 7.], [3., 7.], [4., 7.], [5., 7.], [6., 7.], [7., 7.],
        ];
        let output = generate_grid(8);
        assert_eq!(output, expected);

        // let grid = meshgrid(&array![0, 1, 2], &array![0, 1]);
        // dbg!(grid);
    }
}
