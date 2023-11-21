use crate::utl::{AsArray, *};
use anyhow::Result;
use image::{imageops, Luma};
use imageproc::{
    contours::Contour,
    geometric_transformations::{warp, Interpolation, Projection},
    point::Point,
    *,
};
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use std::cmp::Reverse;

pub fn detect_board(src: &LumaImg<u8>) -> Result<LumaImg<u8>> {
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
    update_contours(&mut contours, &spts);

    // for each contour, treat their points as a perfect square and calculate
    // the projection matrix. Use this projection matrix to generate a set of
    // points around this contour and see how closely they match those
    // previously detected. If the generated points are in proximity of
    // detected saddle points, treat them as a match and update their
    // coordinates to better fit saddle point and repeat process using more
    // points. Keep track of the number of inliers for given projection and
    // find a projection with the biggest number of inliers
    let mut best_projections = vec![];
    let mut inliers_count = 0;

    for cnt in &contours {
        // calculate projection matrix that transforms identity square into given quad
        let mut projection = calc_projection(cnt).as_array();
        let mut inliers;
        // with every iteration, generate new projection matrix that better fits saddle points in the image
        for n in 1..8 {
            // generate a grid containing points from the considered contour in the center
            let (mut transformed_grid, ideal_grid) = chess_grid(&projection, n);

            // snap grid points to closest saddle point within range
            inliers = update_grid(&mut transformed_grid, &spts, 2.);
            inliers_count = inliers.iter().map(|&x| x as u32).sum::<u32>();

            // generate new projection matrix for updated grid
            if let Err(err) =
                update_projection(&mut projection, &ideal_grid, &transformed_grid, &inliers)
            {
                eprintln!("{err}");
                break;
            }

            let points_count = inliers.len();
            println!("iteration {n}: matched {inliers_count} out of {points_count} points");
        }
        best_projections.push((projection, inliers_count));
    }

    // finds the projection with the biggest number of inliers
    best_projections.sort_by_key(|(_, inliers_count)| Reverse(*inliers_count));
    let (best_projection, _) = &best_projections[0];

    // calculates a set of points that surround a chessboard and uses them to
    // find a projection that strechess chessboard to the edges of image
    let offset = 7.;
    let a = best_projection.dot(&array![-offset, -offset, 1.]);
    let b = best_projection.dot(&array![offset, -offset, 1.]);
    let c = best_projection.dot(&array![offset, offset, 1.]);
    let d = best_projection.dot(&array![-offset, offset, 1.]);
    let a = (a[0] / a[2], a[1] / a[2]);
    let b = (b[0] / b[2], b[1] / b[2]);
    let c = (c[0] / c[2], c[1] / c[2]);
    let d = (d[0] / d[2], d[1] / d[2]);
    let projection = Projection::from_control_points(
        [a, b, c, d],
        [(0., 0.), (1280., 0.), (1280., 720.), (0., 720.)],
    )
    .unwrap();

    // transforms image into perfect square with chessboard in the middle
    let res = warp(src, &projection, Interpolation::Nearest, Luma([255]));
    let res = imageops::resize(&res, 1280, 1280, imageops::FilterType::Nearest);

    display(&res);
    Ok(res)
}

/// Calculates image gradients and finds points on the surface of the graph of
/// those functions where the derivatives in orthogonal directions are zero.
fn saddle_points(img: &LumaImg<f32>) -> Array2<f32> {
    let g = Gradients::compute(img);
    let mut saddle = amap!((xx in &g.xx, yy in &g.yy, xy in &g.xy) {
        let s = -(xx * yy - xy.powi(2));
        if s < 0. { 0. } else { s }
    });
    saddle.threshold(1.0);
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
fn update_contours(contours: &mut Vec<Contour<u32>>, spts: &Array1<(usize, usize)>) {
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

/// Translates points inside Contour to an array of tuples
fn to_points<T: Number>(contour: &Contour<T>) -> [(T, T); 4] {
    match contour.points.as_slice() {
        [a, b, c, d] => [a.as_pair(), b.as_pair(), c.as_pair(), d.as_pair()],
        _ => unreachable!(),
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

/// Calculates projection matrix that transforms perfect identity square into a
/// given contour
fn calc_projection(contour: &Contour<u32>) -> Projection {
    let to = to_points(contour).map(|(x, y)| (x as f32, y as f32));
    let from = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)];
    Projection::from_control_points(from, to).unwrap()
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

/// Generates a grid of points with padding for n square chessboard by transforming
/// ideal chessboard with given proojection matrix
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
    (transformed_grid, orthogonal_grid)
}

/// Calcuates which of saddle points is closest to given point
fn min_saddle_distance(
    pt: (f32, f32),
    saddle_points: &Array1<(usize, usize)>,
) -> ((f32, f32), f32) {
    let (mut best_dist, mut best_point) = (f32::MAX, pt);
    for sp in saddle_points {
        let sp = (sp.1 as f32, sp.0 as f32); // saddle points are in y, x order
        let dist = (sp.0 - pt.0).powi(2) + (sp.1 - pt.1).powi(2);
        if dist < best_dist {
            best_dist = dist;
            best_point = sp;
        }
    }
    (best_point, best_dist.sqrt())
}

/// Updates points on a grid by snapping them to a closest saddle point within range
fn update_grid(
    transformed_grid: &mut Array2<f32>,
    saddle_points: &Array1<(usize, usize)>,
    window: f32,
) -> Vec<bool> {
    let mut updated = vec![false; transformed_grid.len_of(Axis(0))];

    for (n, mut arr) in transformed_grid.axis_iter_mut(Axis(0)).enumerate() {
        let pt = (arr[0], arr[1]);
        let (best_point, dist) = min_saddle_distance(pt, saddle_points);
        if dist < window {
            arr[0] = best_point.0;
            arr[1] = best_point.1;
            updated[n] = true;
        }
    }
    updated
}

fn update_projection(
    projection: &mut Array2<f32>,
    ideal_grid: &Array2<f32>,
    transformed_grid: &Array2<f32>,
    inliers: &[bool],
) -> Result<()> {
    let tform_iter = transformed_grid
        .axis_iter(Axis(0))
        .map(|arr| [arr[0] as f64, arr[1] as f64]);
    let ideal_iter = ideal_grid
        .axis_iter(Axis(0))
        .map(|arr| [arr[0] as f64, arr[1] as f64]);
    let matches = tform_iter
        .zip(ideal_iter)
        .enumerate()
        .filter(|(n, _)| inliers[*n])
        .map(|(_, (to, from))| [from, to])
        .collect_vec();

    let homography = super::arrsac(&matches)?;
    let iter = homography.into_iter().map(|x| x as f32);
    *projection = Array::from_iter(iter).into_shape((3, 3))?;
    Ok(())
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
    }
}
