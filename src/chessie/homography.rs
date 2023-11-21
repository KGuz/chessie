use arrsac::Arrsac;
use itertools::Itertools;
use nalgebra::{distance_squared, Const, Matrix3, Point2, SMatrix};
use rand::{seq::SliceRandom, SeedableRng};
use rand_pcg::Pcg64;
use sample_consensus::{Consensus, Estimator, Model};
type Point = [f64; 2];

struct HomographyMatrix([f64; 9]);
impl Model<[Point; 2]> for HomographyMatrix {
    fn residual(&self, data: &[Point; 2]) -> f64 {
        let Self(mat) = self;
        let [a, b] = data;

        let mat = Matrix3::new(
            mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8],
        );
        let (a, b) = (Point2::new(a[0], a[1]), Point2::new(b[0], b[1]));
        let b2 = Point2::from_homogeneous(mat * a.to_homogeneous());

        if let Some(b2) = b2 {
            distance_squared(&b, &b2)
        } else {
            f64::MAX
        }
    }
}

struct HomographyEstimator;
impl Estimator<[Point; 2]> for HomographyEstimator {
    type Model = HomographyMatrix;
    type ModelIter = Option<HomographyMatrix>;
    const MIN_SAMPLES: usize = 4;

    fn estimate<I>(&self, data: I) -> Self::ModelIter
    where
        I: Iterator<Item = [Point; 2]> + Clone,
    {
        let matches = data.take(Self::MIN_SAMPLES).collect_vec();
        homography(&matches).map(HomographyMatrix)
    }
}

pub fn homography(matches: &[[Point; 2]]) -> Option<[f64; 9]> {
    let (m1, m2): (Vec<Point2<f64>>, Vec<Point2<f64>>) = matches
        .iter()
        .map(|m| (Point2::new(m[0][0], m[0][1]), Point2::new(m[1][0], m[1][1])))
        .unzip();

    let count = m1.len();
    let mut c2: Point2<f64> = Point2::origin();
    let mut c1: Point2<f64> = Point2::origin();

    for i in 0..count {
        c2.x += m2[i].x;
        c2.y += m2[i].y;
        c1.x += m1[i].x;
        c1.y += m1[i].y;
    }

    c2.x /= count as f64;
    c2.y /= count as f64;
    c1.x /= count as f64;
    c1.y /= count as f64;

    let mut s2: Point2<f64> = Point2::origin();
    let mut s1: Point2<f64> = Point2::origin();

    for i in 0..count {
        s2.x += (c2.x - m2[i].x).abs();
        s2.y += (c2.y - m2[i].y).abs();
        s1.x += (c1.x - m1[i].x).abs();
        s1.y += (c1.y - m1[i].y).abs();
    }

    if s2.x.abs() < f64::EPSILON
        || s2.y.abs() < f64::EPSILON
        || s1.x.abs() < f64::EPSILON
        || s1.y.abs() < f64::EPSILON
    {
        return None;
    }

    s2.x = count as f64 / s2.x;
    s2.y = count as f64 / s2.y;
    s1.x = count as f64 / s1.x;
    s1.y = count as f64 / s1.y;

    let inv_h_norm = Matrix3::new(1. / s2.x, 0., c2.x, 0., 1. / s2.y, c2.y, 0., 0., 1.);
    let h_norm2 = Matrix3::new(s1.x, 0., -c1.x * s1.x, 0., s1.y, -c1.y * s1.y, 0., 0., 1.);

    let mut ltl: SMatrix<f64, 9, 9> = SMatrix::zeros();
    for i in 0..count {
        let x2 = (m2[i].x - c2.x) * s2.x;
        let y2 = (m2[i].y - c2.y) * s2.y;
        let x1 = (m1[i].x - c1.x) * s1.x;
        let y1 = (m1[i].y - c1.y) * s1.y;
        let lx = [x1, y1, 1., 0., 0., 0., -x2 * x1, -x2 * y1, -x2];
        let ly = [0., 0., 0., x1, y1, 1., -y2 * x1, -y2 * y1, -y2];
        // println!("{} lx {:?} ly {:?}", i, lx, ly);
        for j in 0..9 {
            for k in 0..9 {
                ltl[(j, k)] += lx[j] * lx[k] + ly[j] * ly[k];
            }
        }
    }

    ltl.fill_lower_triangle_with_upper_triangle();
    let eigen = ltl.symmetric_eigen();

    let (eigen_vector_idx, _) = eigen.eigenvalues.argmin();
    let h0 = eigen.eigenvectors.column(eigen_vector_idx);
    let h0 = h0
        .clone_owned()
        .reshape_generic(Const::<3>, Const::<3>)
        .transpose();

    let res = (inv_h_norm * h0) * h_norm2;
    let res = res * (1.0 / res[(2, 2)]);

    let res = res.data.0;
    #[rustfmt::skip]
    let res = [
        res[0][0], res[1][0], res[2][0], 
        res[0][1], res[1][1], res[2][1], 
        res[0][2], res[1][2], res[2][2],
    ];
    Some(res)
}

pub fn arrsac(matches: &[[Point; 2]]) -> Option<[f64; 9]> {
    let mut matches = matches.to_owned();
    matches.shuffle(&mut Pcg64::seed_from_u64(0xDEADC0DE));

    let mut arrsac = Arrsac::new(1., Pcg64::seed_from_u64(0xDEADC0DE));
    let estimator = HomographyEstimator;

    let model = arrsac.model(&estimator, matches.into_iter());
    model.map(|m| m.0)
}
