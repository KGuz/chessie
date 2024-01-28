use super::{homography, types::*};
use image::imageops;
use itertools::izip;
use nalgebra::{point, DMatrix, Matrix3, Point2};

pub fn run(src: &RgbImg<u8>) -> Result<LumaImg<u8>> {
    let uimg = preprocess(src, (1280, 720));
    let fimg = imageops::grayscale_with_type(&uimg);

    // find all the points that can determine the corners of the chess squares
    let mut saddle_points = find_saddle_points(&fimg);
    update_saddle_points(&mut saddle_points, 16, 1.);

    // transform the matrix into a vector of points
    let markers = find_markers(&saddle_points);

    // find all contours inside the image
    let mut quads = find_quads(&uimg);
    update_quads(&mut quads, &markers, 16.);

    // compute a transformation matrix that warps the chessboard to the corners of the image
    let unit_tform = find_unit_square_transform(&quads, &markers);
    let img_tform = calculate_image_transform(&unit_tform, uimg.dimensions())?;

    let res = transform_image(&uimg, &img_tform);
    Ok(res)
}

/// Converts the image to a fixed size and changes the underlying pixel type
pub fn preprocess(src: &RgbImg<u8>, shape: (u32, u32)) -> LumaImg<u8> {
    let img = imageops::resize(src, shape.0, shape.1, imageops::FilterType::Nearest);
    let img = imageops::grayscale(&img);
    imageproc::filter::median_filter(&img, 1, 1)
}

/// Calculates second degree gradients of the image
fn gradients(image: &LumaImg<f32>) -> [LumaImg<f32>; 3] {
    use imageproc::filter;

    let h = [-1., 0., 1., -2., 0., 2., -1., 0., 1.];
    let v = [-1., -2., -1., 0., 0., 0., 1., 2., 1.];

    let x: LumaImg<f32> = filter::filter3x3(image, &h);
    let y: LumaImg<f32> = filter::filter3x3(image, &v);

    let xx: LumaImg<f32> = filter::filter3x3(&x, &h);
    let yy: LumaImg<f32> = filter::filter3x3(&y, &v);
    let xy: LumaImg<f32> = filter::filter3x3(&x, &v);

    [xx, yy, xy]
}

/// Calculates the result of a function: **-(xx · yy - xy<sup>2</sup>)**, whose
/// components are the second-degree gradients of the image. All positive values
/// determine its [saddle points](https://en.wikipedia.org/wiki/Saddle_point)
fn find_saddle_points(image: &LumaImg<f32>) -> DMatrix<f32> {
    let [gxx, gyy, gxy] = gradients(image);
    let iter = izip!(gxx.iter(), gyy.iter(), gxy.iter());

    let f = |(xx, yy, xy): (&f32, &f32, &f32)| -(xx * yy - xy * xy);

    let (width, height) = (image.width() as usize, image.height() as usize);
    DMatrix::from_row_iterator(height, width, iter.map(f))
}

/// Performs a non-max suppression algorithm to eliminate overlapping detections
/// and thresholds the matrix with the average value
fn update_saddle_points(mat: &mut DMatrix<f32>, window: usize, threshold: f32) {
    let (nrows, ncols) = mat.shape();

    let mut nms = DMatrix::zeros(nrows, ncols);
    let (mut sum, mut count) = (0., 0);

    for c in window..ncols - window {
        for r in window..nrows - window {
            let val = mat[(r, c)];
            if val <= threshold {
                continue;
            }

            let [c0, cn] = [c - window, c + window];
            let [r0, rn] = [r - window, r + window];

            let mut max = 0.;
            for ic in c0..=cn {
                for ir in r0..=rn {
                    if max < mat[(ir, ic)] {
                        max = mat[(ir, ic)];
                    }
                }
            }

            if (max - val).abs() < f32::EPSILON {
                nms[(r, c)] = val;
                sum += val;
                count += 1;
            }
        }
    }

    let avg = sum / count as f32;
    nms.apply(|v| *v = if *v < avg { 0. } else { *v });
    *mat = nms;
}

/// Finds all matrix coordinates whose values are greater than zero
pub fn find_markers(mat: &DMatrix<f32>) -> Vec<Point2<usize>> {
    let mut markers = vec![];

    for c in 0..mat.ncols() {
        for r in 0..mat.nrows() {
            if mat[(r, c)] > 0. {
                markers.push(point!(c, r));
            }
        }
    }
    markers
}

/// Detects possible chessboard squares by smoothing the contours obtained from
/// the Canny Edge Detection algorithm. The contours are then verified using
/// several quad-specific heuristics
fn find_quads(img: &LumaImg<u8>) -> Vec<Quad<usize>> {
    use imageproc::{contours, distance_transform::Norm, edges, geometry, morphology};
    let edges = edges::canny(img, 20., 250.);
    let morphed = morphology::dilate(&edges, Norm::L1, 3);
    let contours = contours::find_contours(&morphed);

    contours
        .into_iter()
        .filter_map(|cnt| {
            let arc_len = geometry::arc_length(&cnt.points, true);
            let poly_dp = geometry::approximate_polygon_dp(&cnt.points, 0.05 * arc_len, true);

            let quad = match &poly_dp[..] {
                [a, b, c, d] => [
                    point!(a.x, a.y),
                    point!(b.x, b.y),
                    point!(c.x, c.y),
                    point!(d.x, d.y),
                ],
                _ => return None,
            };

            (cnt.parent.is_some() && polygon_area(&quad) > 64 && is_square_like(&quad, 0.1))
                .then_some(quad)
        })
        .collect()
}

/// Calculates the surface area enclosed by a set of four points
fn polygon_area([a, b, c, d]: &Quad<usize>) -> usize {
    let upper = a.x * b.y + b.x * c.y + c.x * d.y + d.x * a.y;
    let lower = a.y * b.x + b.y * c.x + c.y * d.x + d.y * a.x;
    (upper - lower) / 2
}

/// A heuristic based on the fact that a square should have two pairs of
/// sides of equal length and diagonals of equal length
fn is_square_like(points: &Quad<usize>, epsilon: f32) -> bool {
    use nalgebra::distance as dist;
    let [a, b, c, d] = points.map(Point2::cast::<f32>);

    let sides = (dist(&a, &b), dist(&b, &c), dist(&c, &d), dist(&d, &a));
    let diags = (dist(&a, &c), dist(&b, &d));

    let are_equal = |x1: f32, x2: f32| {
        let avg = 0.5 * (x1 + x2);
        let (e1, e2) = ((x1 - avg).abs() / avg, (x2 - avg).abs() / avg);
        e1 < epsilon && e2 < epsilon
    };

    are_equal(sides.0, sides.2) && are_equal(sides.1, sides.3) && are_equal(diags.0, diags.1)
}

/// Compares points inside the quad with markers and updates them if the
/// distance between them is within a radius
fn update_quads(quads: &mut [Quad<usize>], markers: &[Point2<usize>], radius: f32) {
    let udist_sq = |a: &Point2<usize>, b: &Point2<usize>| {
        let (dx, dy) = (a.x.abs_diff(b.x), a.y.abs_diff(b.y));
        dx * dx + dy * dy
    };

    let radius_sq = (radius * radius) as usize;
    for pt in quads.iter_mut().flatten() {
        let closest = markers.iter().min_by_key(|mk| udist_sq(pt, mk)).unwrap();
        if udist_sq(pt, closest) < radius_sq {
            *pt = *closest;
        }
    }
}

/// Attempts to compute a projection matrix that transforms the identity square
/// into the square defined by the corners of the chessboard
/// ### Explanation
/// For each contour, it computes a projection matrix that transforms a perfect
/// square into a quad defined by the contour points. It then uses this
/// projection matrix to generate a set of points around the contour and checks
/// how closely they correspond to previously detected marker points. If the
/// generated points are nearby, it treats them as a match and updates their
/// coordinates. If all points for a given projection have been successfully
/// correlated, the process is repeated with more points until all adjacent
/// squares are found. To find the best projection, the algorithm keeps track
/// of the number of inliers for each projection and selects the one with the
/// most inliers
fn find_unit_square_transform(quads: &[Quad<usize>], markers: &[Point2<usize>]) -> Matrix3<f64> {
    let square = [
        point!(0., 0.),
        point!(1., 0.),
        point!(1., 1.),
        point!(0., 1.),
    ];

    let (mut projection, mut count) = (Matrix3::zeros(), 0);
    for quad in quads {
        let matches = izip!(square, quad.map(Point2::cast::<f64>)).collect();
        let Ok(mut h) = homography::homography(matches) else {
            continue;
        };

        let mut inliers = vec![];
        for n in 1..5 {
            let orthogonal_grid = generate_grid(2 * (n + 1));

            let mut transformed_grid = transform_grid(&orthogonal_grid, &h);
            inliers = update_grid(&mut transformed_grid, markers, 2.);

            if update_projection(&mut h, &orthogonal_grid, &transformed_grid, &inliers).is_err() {
                break;
            };
        }

        let inliers_count = inliers.iter().filter(|&&n| n).count();
        if count < inliers_count {
            count = inliers_count;
            projection = h;
        }
    }
    projection
}

/// Generates an n by n array of grid points
/// ### Example
/// ```
/// use nalgebra::point;
/// let grid = vec![
///     point!(-1., -1.), point!(0., -1.), point!(1., -1.),
///     point!(-1.,  0.), point!(0.,  0.), point!(1.,  0.),
///     point!(-1.,  1.), point!(0.,  1.), point!(1.,  1.),
/// ];
/// assert_eq!(grid, chessie::generate_grid(3))
/// ```
pub fn generate_grid(n: usize) -> Vec<Point2<f64>> {
    let offset = (n / 2) as f64;
    (0..n * n)
        .map(|i| point!((i % n) as f64 - offset, (i / n) as f64 - offset))
        .collect()
}

/// Transforms a set of points using a projection matrix
fn transform_grid(grid: &[Point2<f64>], proj: &Matrix3<f64>) -> Vec<Point2<f64>> {
    grid.iter()
        .map(|p| Point2::from_homogeneous(proj * p.to_homogeneous()).unwrap())
        .collect()
}

/// Updates points on the grid by snapping them to the nearest marker within radius
fn update_grid(
    transformed_grid: &mut [Point2<f64>],
    markers: &[Point2<usize>],
    radius: f64,
) -> Vec<bool> {
    let mut inliers = vec![false; transformed_grid.len()];
    let radius_sq = radius * radius;

    for (n, pt) in transformed_grid.iter_mut().enumerate() {
        let closest = markers
            .iter()
            .map(|mk| mk.cast())
            .min_by_key(|mk| nalgebra::distance_squared(mk, pt) as usize)
            .expect("markers should not be empty");

        if nalgebra::distance_squared(pt, &closest) < radius_sq {
            *pt = closest;
            inliers[n] = true;
        }
    }
    inliers
}

/// Attempts to update the projection matrix with the one generated by the
/// Adaptive Real-time Random Sample Consensus algorithm
fn update_projection(
    projection: &mut Matrix3<f64>,
    from: &[Point2<f64>],
    to: &[Point2<f64>],
    inliers: &[bool],
) -> Result<()> {
    let matches = izip!(from, to, inliers)
        .filter_map(|(&f, &t, &i)| if i { Some((f, t)) } else { None })
        .collect();

    *projection = homography::arrsac(matches)?;
    Ok(())
}

/// Attempts to calculate a projection matrix that streches chessboard to the
/// edges of the image
fn calculate_image_transform(projection: &Matrix3<f64>, shape: (u32, u32)) -> Result<Matrix3<f64>> {
    let (w, h) = (shape.0 as f64, shape.1 as f64);
    let scale = 6.;

    let transform_point = |x: f64, y: f64| {
        // reverse translation from generate_grid
        let [x, y] = [x, y].map(|v| (2. * v - 1.) * scale);
        Point2::from_homogeneous(projection * point!(x, y).to_homogeneous()).unwrap()
    };

    #[rustfmt::skip]
    let matches = vec![
        (transform_point(0., 0.), point!(0., 0.)),
        (transform_point(1., 0.), point!(w , 0.)),
        (transform_point(1., 1.), point!(w , h )),
        (transform_point(0., 1.), point!(0., h )),
    ];
    homography::homography(matches)
}

/// Warps the image using a transformation matrix and resizes it so that
/// the checkerboard fills the entire image
fn transform_image(image: &LumaImg<u8>, transform: &Matrix3<f64>) -> LumaImg<u8> {
    use imageproc::geometric_transformations::{warp, Interpolation, Projection};

    let arr: [f64; 9] = transform.transpose().as_slice().try_into().unwrap();
    let h = Projection::from_matrix(arr.map(|x| x as f32)).unwrap();

    let size = image.width().max(image.height());

    let res = warp(image, &h, Interpolation::Nearest, image::Luma([255]));
    imageops::resize(&res, size, size, imageops::FilterType::Nearest)
}
