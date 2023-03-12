#![allow(unused)]
use imageproc::point::Point;
use ndarray::{array, Array1, ArrayView1};
use num::NumCast;
use std::fmt::Debug;
use std::ops::*;

#[derive(
    // BitAnd, BitOr, BitXor, Rem, Shr, Shl, Sum, Product, BitAndAssign, BitOrAssign, BitXorAssign, RemAssign, ShrAssign, ShlAssign,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Debug,
    Default,
    Clone,
    Copy,
    Hash,
)]
pub struct Pt<T>(T, T);

macro_rules! pt {
    () => {
        Pt::default()
    };
    ($p: expr) => {{
        Pt::from($p).into_type()
    }};
    ($x: expr, $y: expr) => {
        Pt($x, $y)
    };
}
pub(crate) use pt;

////////////////////////////////// Operators //////////////////////////////////

impl<T: Neg<Output = T>> Neg for Pt<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Pt(-self.0, -self.1)
    }
}

impl<T: Add<Output = T> + Copy> Add<T> for Pt<T> {
    type Output = Self;
    fn add(self, rhs: T) -> Self::Output {
        Pt(self.0 + rhs, self.1 + rhs)
    }
}

impl<T: Add<Output = T>> Add<Pt<T>> for Pt<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Pt(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl<T: AddAssign + Copy> AddAssign<T> for Pt<T> {
    fn add_assign(&mut self, rhs: T) {
        self.0 += rhs;
        self.1 += rhs
    }
}

impl<T: AddAssign> AddAssign<Pt<T>> for Pt<T> {
    fn add_assign(&mut self, rhs: Pt<T>) {
        self.0 += rhs.0;
        self.1 += rhs.1
    }
}

impl<T: Sub<Output = T> + Copy> Sub<T> for Pt<T> {
    type Output = Self;
    fn sub(self, rhs: T) -> Self::Output {
        Pt(self.0 - rhs, self.1 - rhs)
    }
}

impl<T: Sub<Output = T>> Sub<Pt<T>> for Pt<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Pt(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl<T: SubAssign + Copy> SubAssign<T> for Pt<T> {
    fn sub_assign(&mut self, rhs: T) {
        self.0 -= rhs;
        self.1 -= rhs
    }
}

impl<T: SubAssign> SubAssign<Pt<T>> for Pt<T> {
    fn sub_assign(&mut self, rhs: Pt<T>) {
        self.0 -= rhs.0;
        self.1 -= rhs.1
    }
}

impl<T: Mul<Output = T> + Copy> Mul<T> for Pt<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        Pt(self.0 * rhs, self.1 * rhs)
    }
}

impl<T: Mul<Output = T>> Mul<Pt<T>> for Pt<T> {
    type Output = Self;
    fn mul(self, rhs: Pt<T>) -> Self::Output {
        Pt(self.0 * rhs.0, self.1 * rhs.1)
    }
}

impl<T: MulAssign + Copy> MulAssign<T> for Pt<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.0 *= rhs;
        self.1 *= rhs
    }
}

impl<T: MulAssign> MulAssign<Pt<T>> for Pt<T> {
    fn mul_assign(&mut self, rhs: Pt<T>) {
        self.0 *= rhs.0;
        self.1 *= rhs.1
    }
}

impl<T: Debug + num::Zero + PartialOrd + Div<Output = T> + Copy> Div<T> for Pt<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        assert_ne!(rhs, T::zero());
        Pt(self.0 / rhs, self.1 / rhs)
    }
}

impl<T: Debug + num::Zero + PartialOrd + Div<Output = T>> Div<Pt<T>> for Pt<T> {
    type Output = Self;
    fn div(self, rhs: Pt<T>) -> Self::Output {
        assert_ne!(rhs.0, T::zero());
        assert_ne!(rhs.1, T::zero());
        Pt(self.0 / rhs.0, self.1 / rhs.1)
    }
}

impl<T: Debug + num::Zero + PartialOrd + DivAssign + Copy> DivAssign<T> for Pt<T> {
    fn div_assign(&mut self, rhs: T) {
        assert_ne!(rhs, T::zero());
        self.0 /= rhs;
        self.1 /= rhs
    }
}

impl<T: Debug + num::Zero + PartialOrd + DivAssign> DivAssign<Pt<T>> for Pt<T> {
    fn div_assign(&mut self, rhs: Pt<T>) {
        assert_ne!(rhs.0, T::zero());
        assert_ne!(rhs.1, T::zero());
        self.0 /= rhs.0;
        self.1 /= rhs.1
    }
}

///////////////////////////////// Conversions /////////////////////////////////

impl<T> From<Point<T>> for Pt<T> {
    fn from(point: Point<T>) -> Self {
        Pt(point.x, point.y)
    }
}

impl<T: Copy> From<&Point<T>> for Pt<T> {
    fn from(point: &Point<T>) -> Self {
        Pt(point.x, point.y)
    }
}

impl<T> From<(T, T)> for Pt<T> {
    fn from(tuple: (T, T)) -> Self {
        Pt(tuple.0, tuple.1)
    }
}

impl<T: Copy> From<&(T, T)> for Pt<T> {
    fn from(tuple: &(T, T)) -> Self {
        Pt(tuple.0, tuple.1)
    }
}

impl<T: Copy> From<(&T, &T)> for Pt<T> {
    fn from(tuple: (&T, &T)) -> Self {
        Pt(*tuple.0, *tuple.1)
    }
}

impl<T> From<[T; 2]> for Pt<T> {
    fn from(array: [T; 2]) -> Self {
        let [x, y] = array;
        Pt(x, y)
    }
}

impl<T: Copy> From<&[T; 2]> for Pt<T> {
    fn from(array: &[T; 2]) -> Self {
        let &[x, y] = array;
        Pt(x, y)
    }
}

impl<T: Copy> From<[&T; 2]> for Pt<T> {
    fn from(array: [&T; 2]) -> Self {
        let [&x, &y] = array;
        Pt(x, y)
    }
}

impl<T> From<Pt<T>> for (T, T) {
    fn from(pt: Pt<T>) -> Self {
        (pt.0, pt.1)
    }
}

impl<T: Copy> From<&Pt<T>> for (T, T) {
    fn from(pt: &Pt<T>) -> Self {
        (pt.0, pt.1)
    }
}

impl<T> From<Pt<T>> for [T; 2] {
    fn from(pt: Pt<T>) -> Self {
        [pt.0, pt.1]
    }
}

impl<T: Copy> From<&Pt<T>> for [T; 2] {
    fn from(pt: &Pt<T>) -> Self {
        [pt.0, pt.1]
    }
}

impl<T> From<Vec<T>> for Pt<T> {
    fn from(vec: Vec<T>) -> Self {
        assert_eq!(vec.len(), 2, "Invalid length of vector");
        let mut it = vec.into_iter().take(2);
        Pt(it.next().unwrap(), it.next().unwrap())
    }
}

impl<T: Copy> From<&[T]> for Pt<T> {
    fn from(slice: &[T]) -> Self {
        assert_eq!(slice.len(), 2, "Invalid length of slice");
        let mut it = slice.iter().take(2);
        Pt(*it.next().unwrap(), *it.next().unwrap())
    }
}

impl<T> From<Array1<T>> for Pt<T> {
    fn from(array1: Array1<T>) -> Self {
        assert_eq!(array1.len(), 2, "Invalid length of ndarray");
        let mut it = array1.into_iter().take(2);
        Pt(it.next().unwrap(), it.next().unwrap())
    }
}

impl<T: Copy> From<&Array1<T>> for Pt<T> {
    fn from(array1: &Array1<T>) -> Self {
        assert_eq!(array1.len(), 2, "Invalid length of ndarray");
        let mut it = array1.into_iter().take(2);
        Pt(*it.next().unwrap(), *it.next().unwrap())
    }
}

impl<'a, T: Copy> From<ArrayView1<'a, T>> for Pt<T> {
    fn from(array1: ArrayView1<T>) -> Self {
        assert_eq!(array1.len(), 2, "Invalid length of ndarray view");
        let mut it = array1.into_iter().take(2);
        Pt(*it.next().unwrap(), *it.next().unwrap())
    }
}

impl<T: Copy + NumCast> Pt<T> {
    pub fn into_type<U: NumCast>(self) -> Pt<U> {
        Pt(num::cast(self.0).unwrap(), num::cast(self.1).unwrap())
    }

    pub fn as_type<U: NumCast>(&self) -> Pt<U> {
        Pt(num::cast(self.0).unwrap(), num::cast(self.1).unwrap())
    }
}

////////////////////////////////// Iterators //////////////////////////////////

impl<T> IntoIterator for Pt<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, 2>;

    fn into_iter(self) -> Self::IntoIter {
        [self.0, self.1].into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Pt<T> {
    type Item = &'a T;
    type IntoIter = std::array::IntoIter<&'a T, 2>;

    fn into_iter(self) -> Self::IntoIter {
        [&self.0, &self.1].into_iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Pt<T> {
    type Item = &'a mut T;
    type IntoIter = std::array::IntoIter<&'a mut T, 2>;

    fn into_iter(self) -> Self::IntoIter {
        [&mut self.0, &mut self.1].into_iter()
    }
}

impl<T> Pt<T> {
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        [&self.0, &self.1].into_iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        [&mut self.0, &mut self.1].into_iter()
    }
}

/////////////////////////////// Utility methods ///////////////////////////////

impl<T> Index<usize> for Pt<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.0,
            1 => &self.1,
            _ => panic!("index '{}' is out of bounds of Pt object", index),
        }
    }
}

impl<T> Index<char> for Pt<T> {
    type Output = T;
    fn index(&self, index: char) -> &Self::Output {
        match index {
            'x' => &self.0,
            'y' => &self.1,
            _ => panic!("index '{}' is out of bounds of Pt object", index),
        }
    }
}

impl<T> Pt<T> {
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.0, &mut self.1);
    }
    pub fn map<U>(&self, f: fn(&T) -> U) -> Pt<U> {
        Pt(f(&self.0), f(&self.1))
    }
    pub fn apply(&mut self, f: fn(&mut T)) {
        f(&mut self.0);
        f(&mut self.1);
    }
}

impl<T: Copy> Pt<T> {
    pub fn mapv<U>(&self, f: fn(T) -> U) -> Pt<U> {
        Pt(f(self.0), f(self.1))
    }
    pub fn applyv(&mut self, f: fn(T) -> T) {
        *self = Pt(f(self.0), f(self.1));
    }
}

impl<T: Copy + NumCast> Pt<T> {
    pub fn dist(&self, other: &Pt<T>) -> f64 {
        let (p1, p2): (Pt<f64>, Pt<f64>) = (self.as_type(), other.as_type());
        (p2 - p1).mag()
    }
    pub fn mag(&self) -> f64 {
        let p: Pt<f64> = self.as_type();
        (p * p).sum().sqrt()
    }
    // pub fn phase
    pub fn norm(&self) -> Pt<f64> {
        self.as_type() / self.mag()
    }
}

impl<T: PartialOrd> Pt<T> {
    pub fn max(&self) -> &T {
        if self.0 > self.1 {
            &self.0
        } else {
            &self.1
        }
    }
    pub fn min(&self) -> &T {
        if self.0 < self.1 {
            &self.0
        } else {
            &self.1
        }
    }
}

impl<T: Copy + Add<Output = T>> Pt<T> {
    pub fn sum(&self) -> T {
        self.0 + self.1
    }
}
impl<T: Copy + Mul<Output = T>> Pt<T> {
    pub fn product(&self) -> T {
        self.0 * self.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_conversions() {
        let point = Point::new(0, 0);
        let tuple = (0, 0);
        let array = [0, 0];
        let vec = vec![0, 0];
        let slice = &vec[..];
        let array1 = array![0, 0];
        let array1_ref = &array1;
        let array1_view = array1.view();
    }
}
