use imageproc::point::Point;

pub trait IntoPair<A> {
    fn into_pair(self) -> (A, A);
}
pub trait AsPair<A> {
    fn as_pair(&self) -> (A, A);
}

pub trait IntoPoint<A> {
    fn into_point(self) -> Point<A>;
}
pub trait AsPoint<A> {
    fn as_point(&self) -> Point<A>;
}

impl<A> IntoPair<A> for Point<A> {
    fn into_pair(self) -> (A, A) {
        (self.x, self.y)
    }
}

impl<A> IntoPair<A> for [A; 2] {
    fn into_pair(self) -> (A, A) {
        let [x, y] = self;
        (x, y)
    }
}

impl<A: Copy> AsPair<A> for Point<A> {
    fn as_pair(&self) -> (A, A) {
        (self.x, self.y)
    }
}

impl<A: Clone> AsPair<A> for [A; 2] {
    fn as_pair(&self) -> (A, A) {
        (self[0].clone(), self[1].clone())
    }
}

impl<A: Clone> AsPair<A> for Vec<A> {
    fn as_pair(&self) -> (A, A) {
        assert!(self.len() == 2);
        (self[0].clone(), self[1].clone())
    }
}

impl<A: Clone> AsPair<A> for &[A] {
    fn as_pair(&self) -> (A, A) {
        assert!(self.len() == 2);
        (self[0].clone(), self[1].clone())
    }
}

impl<A> IntoPoint<A> for [A; 2] {
    fn into_point(self) -> Point<A> {
        let [x, y] = self;
        Point { x, y }
    }
}

impl<A: Copy> AsPoint<A> for [A; 2] {
    fn as_point(&self) -> Point<A> {
        let &[x, y] = self;
        Point { x, y }
    }
}

impl<A> IntoPoint<A> for (A, A) {
    fn into_point(self) -> Point<A> {
        let (x, y) = self;
        Point { x, y }
    }
}

impl<A: Copy> AsPoint<A> for (A, A) {
    fn as_point(&self) -> Point<A> {
        let &(x, y) = self;
        Point { x, y }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_into_pair() {
        assert_eq!((0, 0), Point { x: 0, y: 0 }.into_pair());
    }

    #[test]
    fn arr_into_pair() {
        assert_eq!((5., 6.), [5., 6.].into_pair());
    }

    #[test]
    fn point_as_pair() {
        assert_eq!((1, 2), Point { x: 1, y: 2 }.as_pair());
    }

    #[test]
    fn arr_as_pair() {
        assert_eq!((5., 6.), [5., 6.].as_pair());
    }

    #[test]
    fn vec_as_pair() {
        assert_eq!((-3, -4), vec![-3, -4].as_pair());
    }

    #[test]
    fn slice_as_pair() {
        assert_eq!((-7, -8), [-7, -8].as_slice().as_pair());
    }

    ///
    #[test]
    fn pair_into_point() {
        assert_eq!(Point::new(0, 0), (0, 0).into_point());
    }

    #[test]
    fn arr_into_point() {
        assert_eq!(Point::new(5., 6.), [5., 6.].into_point());
    }

    #[test]
    fn pair_as_point() {
        assert_eq!(Point::new(1, 2), (1, 2).as_point());
    }

    #[test]
    fn arr_as_point() {
        assert_eq!(Point::new(5., 6.), [5., 6.].as_point());
    }
}
