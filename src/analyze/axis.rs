use itertools::Itertools;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::ops::{Add, Mul, Sub};

/// A single axis of a symbolic shape, represented as a multivariate integer-coefficient polynomial of the relevant unknowns
/// The polynomial is represented as a hashmap from exponent values to coefficients. For example, an entry of `[1, 2] -> 3` represents the term `3x₀x₁²` in the polynomial.
// #[derive(Clone, Debug)]
#[derive(Clone)]
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
            if *v != 0 {
                map.insert(SmallVec::new(), *v);
            }
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

    /// If the entire value consists of only a single variable with coefficient 1, return the index of that variable
    pub fn single_var(&self) -> Option<usize> {
        let Axis::Var(map) = self else {
            return None;
        };
        let mut idx = None;
        for (exps, coef) in map {
            if *coef == 0 {
                continue;
            }
            if *coef != 1 {
                return None;
            }
            if exps.iter().sum::<usize>() == 1 && idx.is_none() {
                idx = Some(exps.iter().find_position(|exp| **exp == 1)?.0)
            } else {
                return None;
            }
        }
        idx
    }

    /// Heuristic for deciding which of two equal expressions to proceed with
    pub fn complexity(&self) -> usize {
        match self {
            Axis::Const(_) => 0,
            Axis::Var(map) => {
                // TODO: Figure out a heuristic for non-constant terms
                1
            }
        }
    }

    /// The number of variabe IDs this `Axis` uses
    pub fn to_nvars(&self) -> usize {
        match self {
            Axis::Const(_) => 0,
            Axis::Var(map) => map.keys().map(SmallVec::len).max().unwrap_or_default(),
        }
    }

    pub fn pow(&self, pow: usize) -> Axis {
        std::iter::repeat_n(self, pow).fold(1.into(), Axis::mul)
    }

    /// Given pairs of variable indices and Axis instances, return the result of substituting each Axis as the given variable
    /// Requires a substitution to be provided for *every* variable, erroring if insufficient substitutions are provided
    pub fn substitute(&self, substs: &HashMap<usize, Axis>) -> anyhow::Result<Axis> {
        use anyhow::Context;

        let substs = (0..self.to_nvars())
            .map(|i| substs.get(&i))
            .collect::<Option<Vec<_>>>()
            .context("Not enough variables provided for substitution")?;

        let Axis::Var(map) = self else {
            // There are no substitutions to be made on a constant
            return Ok(self.clone());
        };

        Ok(map
            .iter()
            .map(|(exps, coef)| {
                Axis::from(*coef)
                    * exps
                        .iter()
                        .enumerate()
                        .filter(|(_, exp)| **exp != 0)
                        .map(|(i, exp)| substs[i].pow(*exp))
                        .product::<Axis>()
            })
            .sum())
    }
}

impl std::fmt::Debug for Axis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::fmt::Display for Axis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const SUBSCRIPT_DIGITS: [char; 10] = uiua::SUBSCRIPT_DIGITS;
        const SUPERSCRIPT_DIGITS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
        /// Turn a number into a subscript or superscript
        fn sub_sup(mut val: usize, charset: [char; 10]) -> String {
            let mut chars = Vec::new();
            if val == 0 {
                return charset[0].into();
            }
            while val > 0 {
                chars.push(charset[val % 10]);
                val /= 10;
            }
            chars.into_iter().rev().collect()
        }

        match self {
            Axis::Const(val) => write!(f, "{val}"),
            Axis::Var(map) => {
                // Whether the first nonzero term has been written yet
                // Used to omit a leading `+`
                let mut begun = false;
                // For each term in the polynomial
                for (exps, coef) in map {
                    if *coef == 0 {
                        continue;
                    }
                    if *coef < 0 {
                        write!(f, "-")?;
                    }
                    // Don't put a `+` before the first term
                    else if begun {
                        write!(f, "+")?;
                    }
                    begun = true;

                    // Coefficient of the term
                    let coef_abs = coef.abs();
                    if coef_abs != 1 {
                        write!(f, "{coef_abs}")?;
                    }

                    // Write each variable in the term as "x" followed by a subscript and a superscript
                    for (exp_i, &exp) in exps.iter().enumerate() {
                        if exp == 0 {
                            continue;
                        }
                        let sub_s = sub_sup(exp_i, SUBSCRIPT_DIGITS);
                        let sup_s = if exp == 1 {
                            String::new()
                        } else {
                            sub_sup(exp, SUPERSCRIPT_DIGITS)
                        };
                        write!(f, "x{sub_s}{sup_s}")?;
                    }
                }
                Ok(())
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
            impl From<&$t> for Axis {
                fn from(value: &$t) -> Axis {
                    Axis::from(*value)
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
                    .fold(HashMap::new(), |mut map, (exps, coef)| {
                        *map.entry(exps).or_default() += coef;
                        map
                    }),
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

impl std::iter::Sum for Axis {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(Add::add).unwrap_or_else(|| Axis::Const(0))
    }
}
impl<'a> std::iter::Sum<&'a Axis> for Axis {
    fn sum<I: Iterator<Item = &'a Axis>>(iter: I) -> Axis {
        iter.fold(Axis::Const(0), |acc, next| acc + next)
    }
}

impl std::iter::Product for Axis {
    fn product<I: Iterator<Item = Axis>>(iter: I) -> Axis {
        iter.reduce(Mul::mul).unwrap_or_else(|| Axis::Const(1))
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

impl std::fmt::Display for Relation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sym = match (self.ineq, self.inv) {
            (false, false) => '=',
            (false, true) => '≠',
            (true, false) => '>',
            (true, true) => '≤',
        };

        write!(f, "{} {} 0", self.expr, sym)
    }
}

#[derive(Clone, Debug)]
pub enum Condition {
    Or(SmallVec<[Relation; 4]>),
}

impl From<Relation> for Condition {
    fn from(value: Relation) -> Self {
        Self::Or(smallvec::smallvec![value])
    }
}

impl Condition {
    /// Returns `Some(false)` if the condition can be determined to be false by only comparing constants
    /// Returns `Some(true)` if the condition can be determined to be true by only comparing constants
    /// Returns `None` otherwise
    pub fn trivial(&self) -> Option<bool> {
        match self {
            Self::Or(rels) => {
                let mut out = Some(false);
                for rel in rels {
                    match rel.trivial() {
                        Some(true) => return Some(true),
                        None => out = None,
                        _ => {}
                    }
                }
                out
            }
        }
    }

    /// The number of variabe IDs this `Condition`'s `Axis` instances use
    pub fn to_nvars(&self) -> usize {
        match self {
            Self::Or(rels) => rels
                .iter()
                .map(|rel| rel.expr.to_nvars())
                .max()
                .unwrap_or_default(),
        }
    }

    pub fn substitute(&self, substs: &HashMap<usize, Axis>) -> anyhow::Result<Self> {
        match self {
            Self::Or(rels) => rels
                .iter()
                .map(|rel| {
                    let mut rel = rel.clone();
                    rel.expr = rel.expr.substitute(substs)?;
                    Ok(rel)
                })
                .collect::<anyhow::Result<SmallVec<[Relation; 4]>>>()
                .map(Self::Or),
        }
    }
}

impl std::fmt::Display for Condition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Condition::Or(rels) => {
                if rels.is_empty() {
                    write!(f, "FALSE")?;
                } else {
                    write!(f, "{}", rels[0])?;
                    for rel in &rels[1..] {
                        write!(f, " OR {}", rel)?;
                    }
                }
            }
        }

        Ok(())
    }
}
