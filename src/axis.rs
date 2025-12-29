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
                                .zip(rexps)
                                .map(|(l, r)| *l + *r)
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
