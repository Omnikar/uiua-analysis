use itertools::Itertools;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::ops::{Add, Mul, Sub};

/// A single axis of a symbolic shape, represented as a multivariate integer-coefficient polynomial of the relevant unknowns
/// The polynomial is represented as a hashmap from exponent values to coefficients. For example, an entry of `[1,2] -> 3` represents the term `3 * x0 * x1^2` in the polynomial.
#[derive(Clone, Debug)]
pub enum Axis {
    Const(isize),
    Var(HashMap<SmallVec<[usize; 4]>, isize>),
}

impl Axis {
    /// Given a variable counter, make a new variable with the next unoccupied index and increment the counter
    pub fn newvar(nvars: &mut usize) -> Axis {
        let mut idx = smallvec::smallvec![0; *nvars];
        idx.push(1);
        let mut map = HashMap::new();
        map.insert(idx, 1);
        *nvars += 1;
        Axis::Var(map)
    }

    /// Get the coefficient of the term corresponding to the given set of exponents
    pub fn term(&self, mut exps: &[usize]) -> isize {
        while let Some(trimmed) = exps.strip_suffix(&[0]) {
            exps = trimmed;
        }
        match (self, exps) {
            (Axis::Const(v), &[]) => *v,
            (Axis::Var(map), exps) => map.get(exps).copied().unwrap_or_default(),
            _ => 0,
        }
    }

    /// Get a mutable reference to the coefficient of the term corresponding to the given set of exponents
    pub fn term_mut(&mut self, mut exps: &[usize]) -> &mut isize {
        while let Some(trimmed) = exps.strip_suffix(&[0]) {
            exps = trimmed;
        }
        if !exps.is_empty()
            && let Axis::Const(v) = self
        {
            let mut map = HashMap::new();
            map.insert(SmallVec::new(), *v);
            *self = Axis::Var(map);
        }
        match (self, exps) {
            (Axis::Const(v), &[]) => v,
            (Axis::Var(map), exps) => {
                let exps = exps.into();
                map.entry(exps).or_default()
            }
            _ => unreachable!(),
        }
    }

    /// Get the value of the constant term
    pub fn constant(&self) -> isize {
        self.term(&[])
    }

    /// If the entire value is a constant, return that constant
    pub fn only_const(&self) -> Option<isize> {
        match self {
            Axis::Const(v) => Some(*v),
            Axis::Var(map) => {
                let mut out = 0;
                for (exps, coef) in map {
                    if exps.is_empty() {
                        out = *coef;
                    } else if *coef != 0 {
                        return None;
                    }
                }
                Some(out)
            }
        }
    }
}

macro_rules! impl_from_integer {
    ($($t:ty),+) => {
        $(
            impl From<$t> for Axis {
                fn from(value: $t) -> Axis {
                    Axis::Const(value as isize)
                }
            }
        )+
    };
}

impl_from_integer!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize);

impl From<&Axis> for Axis {
    fn from(value: &Axis) -> Axis {
        value.clone()
    }
}

impl Add<&Axis> for Axis {
    type Output = Axis;
    fn add(mut self, rhs: &Axis) -> Axis {
        match rhs {
            Axis::Const(v) => *self.term_mut(&[]) += v,
            #[allow(clippy::suspicious_arithmetic_impl)]
            Axis::Var(map) => {
                let mut is_constant = true;
                map.iter().for_each(|(exps, coef)| {
                    let ptr = self.term_mut(exps);
                    *ptr += coef;
                    is_constant &= *ptr == 0 || exps.is_empty();
                });
                if is_constant {
                    let constant = self.term(&[]);
                    self = Axis::Const(constant);
                }
            }
        }
        self
    }
}
impl Add<Axis> for &Axis {
    type Output = Axis;
    fn add(self, rhs: Axis) -> Axis {
        rhs + self
    }
}
impl Add for Axis {
    type Output = Axis;
    fn add(self, rhs: Axis) -> Axis {
        self + &rhs
    }
}
impl Add for &Axis {
    type Output = Axis;
    fn add(self, rhs: &Axis) -> Axis {
        self.clone() + rhs
    }
}

impl Sub<&Axis> for Axis {
    type Output = Axis;
    fn sub(mut self, rhs: &Axis) -> Axis {
        match rhs {
            Axis::Const(v) => *self.term_mut(&[]) -= v,
            #[allow(clippy::suspicious_arithmetic_impl)]
            Axis::Var(map) => {
                let mut is_constant = true;
                map.iter().for_each(|(exps, coef)| {
                    let ptr = self.term_mut(exps);
                    *ptr -= coef;
                    is_constant &= *ptr == 0 || exps.is_empty();
                });
                if is_constant {
                    let constant = self.term(&[]);
                    self = Axis::Const(constant);
                }
            }
        }
        self
    }
}
impl Sub<Axis> for &Axis {
    type Output = Axis;
    fn sub(self, rhs: Axis) -> Axis {
        rhs - self
    }
}
impl Sub for Axis {
    type Output = Axis;
    fn sub(self, rhs: Axis) -> Axis {
        self - &rhs
    }
}
impl Sub for &Axis {
    type Output = Axis;
    fn sub(self, rhs: &Axis) -> Axis {
        self.clone() - rhs
    }
}

impl Mul for Axis {
    type Output = Axis;
    fn mul(self, rhs: Axis) -> Axis {
        match (self, rhs) {
            (Axis::Const(lhs), Axis::Const(rhs)) => Axis::Const(lhs * rhs),
            (Axis::Const(coef), Axis::Var(mut map)) | (Axis::Var(mut map), Axis::Const(coef)) => {
                map.values_mut().for_each(|v| *v *= coef);
                Axis::Var(map)
            }
            #[allow(clippy::suspicious_arithmetic_impl)]
            (Axis::Var(lhs), Axis::Var(rhs)) => Axis::Var(
                lhs.into_iter()
                    .cartesian_product(rhs.iter())
                    .map(|((lexps, lcoef), (rexps, rcoef))| {
                        (
                            lexps
                                .iter()
                                .copied()
                                .zip_longest(rexps.iter().copied())
                                .map(|eob| eob.or_default())
                                .map(|(l, r)| l + r)
                                .collect::<SmallVec<[usize; 4]>>(),
                            lcoef * *rcoef,
                        )
                    })
                    .collect(),
            ),
        }
    }
}
impl Mul<&Axis> for Axis {
    type Output = Axis;
    fn mul(self, rhs: &Axis) -> Axis {
        self * rhs.clone()
    }
}
impl Mul<Axis> for &Axis {
    type Output = Axis;
    fn mul(self, rhs: Axis) -> Axis {
        self.clone() * rhs
    }
}
impl Mul for &Axis {
    type Output = Axis;
    fn mul(self, rhs: &Axis) -> Axis {
        self.clone() * rhs.clone()
    }
}

impl std::iter::Product for Axis {
    fn product<I: Iterator<Item = Axis>>(iter: I) -> Axis {
        iter.reduce(Mul::mul).unwrap_or(Axis::Const(1))
    }
}
impl<'a> std::iter::Product<&'a Axis> for Axis {
    fn product<I: Iterator<Item = &'a Axis>>(iter: I) -> Axis {
        iter.fold(Axis::Const(1), |acc, next| acc * next)
    }
}

#[derive(Clone, Debug)]
pub struct Relation {
    /// Axis term that must be equal to zero (for `ineq = false`) or greater than zero (for `ineq = true`) for the relation to be satisfied
    pub expr: Axis,
    /// Whether this relation is an inequality
    /// If `false`, this is an equality relationship
    /// If `true`, this is a less-than relationship
    pub ineq: bool,
    /// Whether this relationship is inverted
    /// Being set to `true` turns "equals" into "not equal" and "less-than" into "greater or equal"
    pub inv: bool,
}

impl Relation {
    /// lhs = rhs
    pub fn eq(lhs: impl Into<Axis>, rhs: impl Into<Axis>) -> Self {
        Self {
            expr: rhs.into() - lhs.into(),
            ineq: false,
            inv: false,
        }
    }

    /// lhs ≠ rhs
    pub fn ne(lhs: impl Into<Axis>, rhs: impl Into<Axis>) -> Self {
        Self {
            expr: rhs.into() - lhs.into(),
            ineq: false,
            inv: true,
        }
    }

    /// lhs < rhs
    pub fn lt(lhs: impl Into<Axis>, rhs: impl Into<Axis>) -> Self {
        Self {
            expr: rhs.into() - lhs.into(),
            ineq: true,
            inv: false,
        }
    }

    /// lhs > rhs
    pub fn gt(lhs: impl Into<Axis>, rhs: impl Into<Axis>) -> Self {
        Self {
            expr: lhs.into() - rhs.into(),
            ineq: true,
            inv: false,
        }
    }

    /// lhs ≤ rhs
    pub fn le(lhs: impl Into<Axis>, rhs: impl Into<Axis>) -> Self {
        Self {
            expr: lhs.into() - rhs.into(),
            ineq: true,
            inv: true,
        }
    }

    /// lhs ≥ rhs
    pub fn ge(lhs: impl Into<Axis>, rhs: impl Into<Axis>) -> Self {
        Self {
            expr: rhs.into() - lhs.into(),
            ineq: true,
            inv: true,
        }
    }

    /// Returns `None` if the relation involves any variables
    /// Returns `Some(false)` if the relation involves only constants and is trivially false
    /// Returns `Some(true)` if the relation involves only constants and is trivially true
    pub fn trivial(&self) -> Option<bool> {
        self.expr
            .only_const()
            .map(|val| if self.ineq { val > 0 } else { val == 0 } ^ self.inv)
    }
}
