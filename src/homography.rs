use super::types::Result;
use arrsac::Arrsac;
use nalgebra::{distance_squared, matrix, vector, Matrix3, Point2, SVD};
use rand::{seq::SliceRandom, SeedableRng};
use sample_consensus::{Consensus, Estimator, Model};

type FeatureMatch = (Point2<f64>, Point2<f64>);

struct HomographyMatrix(Matrix3<f64>);
impl Model<FeatureMatch> for HomographyMatrix {
    fn residual(&self, (from, to): &FeatureMatch) -> f64 {
        Point2::from_homogeneous(self.0 * from.to_homogeneous())
            .map_or(f64::MAX, |pt| distance_squared(to, &pt))
    }
}

struct HomographyEstimator;
impl Estimator<FeatureMatch> for HomographyEstimator {
    type Model = HomographyMatrix;
    type ModelIter = Result<HomographyMatrix>;
    const MIN_SAMPLES: usize = 4;

    fn estimate<I>(&self, data: I) -> Self::ModelIter
    where
        I: Iterator<Item = FeatureMatch> + Clone,
    {
        let matches = data.take(Self::MIN_SAMPLES).collect();
        homography(matches).map(HomographyMatrix)
    }
}

/// Attempts to calculate a projection matrix from pairs of points containing
/// outliers using Adaptive Real-time Random Sample Consensus algorithm
pub fn arrsac(mut matches: Vec<FeatureMatch>) -> Result<Matrix3<f64>> {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xDEADC0DE);
    matches.shuffle(&mut rng);

    let model = Arrsac::new(1., rng)
        .model(&HomographyEstimator, matches.into_iter())
        .map(|HomographyMatrix(m)| m);

    model.ok_or("Err: No valid model could be found for the matches")
}

/// Attempts to calculate a projection matrix from four pairs of points
pub fn homography(matches: Vec<FeatureMatch>) -> Result<Matrix3<f64>> {
    #[rustfmt::skip]
    let [xf1, yf1, xf2, yf2, xf3, yf3, xf4, yf4] = [
        matches[0].0.x, matches[0].0.y,
        matches[1].0.x, matches[1].0.y,
        matches[2].0.x, matches[2].0.y,
        matches[3].0.x, matches[3].0.y,
    ];

    #[rustfmt::skip]
    let [xt1, yt1, xt2, yt2, xt3, yt3, xt4, yt4] = [
        matches[0].1.x, matches[0].1.y,
        matches[1].1.x, matches[1].1.y,
        matches[2].1.x, matches[2].1.y,
        matches[3].1.x, matches[3].1.y,
    ];

    #[rustfmt::skip]
    let a = matrix![
        0.0, 0.0, 0.0, -xf1, -yf1, -1.0,  yt1 * xf1,  yt1 * yf1;
        xf1, yf1, 1.0,  0.0,  0.0,  0.0, -xt1 * xf1, -xt1 * yf1;
        0.0, 0.0, 0.0, -xf2, -yf2, -1.0,  yt2 * xf2,  yt2 * yf2;
        xf2, yf2, 1.0,  0.0,  0.0,  0.0, -xt2 * xf2, -xt2 * yf2;
        0.0, 0.0, 0.0, -xf3, -yf3, -1.0,  yt3 * xf3,  yt3 * yf3;
        xf3, yf3, 1.0,  0.0,  0.0,  0.0, -xt3 * xf3, -xt3 * yf3;
        0.0, 0.0, 0.0, -xf4, -yf4, -1.0,  yt4 * xf4,  yt4 * yf4;
        xf4, yf4, 1.0,  0.0,  0.0,  0.0, -xt4 * xf4, -xt4 * yf4;
    ];

    let b = vector!(-yt1, xt1, -yt2, xt2, -yt3, xt3, -yt4, xt4);

    let svd = SVD::try_new(a, true, true, f64::EPSILON, 0)
        .ok_or("Err: Could not compute the Singular Value Decomposition matrix")?;

    svd.solve(&b, f64::EPSILON)
        .map(|h| matrix![h[0], h[1], h[2]; h[3], h[4], h[5]; h[6], h[7], 1.0])
}
