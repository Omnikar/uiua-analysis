//! Primitive-specific functions for propagating static analysis `Info`

use anyhow::{bail, Context, Result};
use itertools::Either;
use smallvec::{smallvec, SmallVec};
use uiua::{SigNode, Value};

use super::axis::{Axis, Condition, Relation};
use super::{analyze_subgraph, Info, ShapeInfo, SymShape};
use crate::graph::DataGraph;

use ShapeInfo::*;

pub struct AnalyzeCtx<'a, 'b, 'c> {
    pub dep_infos: Vec<Info>,
    pub nvars: &'a mut usize,
    pub reqs: &'b mut Vec<Condition>,
    pub uiua: &'c uiua::Uiua,
}

fn n_args<const N: usize>(dep_infos: Vec<Info>) -> Result<[Info; N]> {
    dep_infos.try_into().ok().context("Incorrect arg count")
}
fn one_arg(mut dep_infos: Vec<Info>) -> Result<Info> {
    dep_infos.pop().context("Incorrect arg count")
}
fn two_args(dep_infos: Vec<Info>) -> Result<[Info; 2]> {
    n_args::<2>(dep_infos)
}

fn pervade_shapes(
    shapes: impl IntoIterator<Item = SymShape>,
    reqs: &mut Vec<Condition>,
) -> Result<SymShape> {
    let mut shapes: SmallVec<[SymShape; 4]> = shapes.into_iter().collect();
    for shape in &mut shapes {
        shape.reverse();
    }
    let mut new_shape = SymShape::new();
    let rank: usize = shapes
        .iter()
        .map(SmallVec::len)
        .max()
        .context("Cannot pervade zero shapes")?;
    for _ in 0..rank {
        let new = shapes
            .iter_mut()
            .filter_map(|sh| sh.pop())
            .try_fold(1.into(), |lhs, rhs| match_axes(lhs, rhs, reqs))?;
        new_shape.push(new);
    }
    Ok(new_shape)
}

/// Attempt to match two axis lengths together
fn match_axes(lhs: Axis, rhs: Axis, reqs: &mut Vec<Condition>) -> Result<Axis> {
    // TODO: Currently only supports ahead-of-time fixing
    //       Not sure how to properly address the general case
    if lhs.only_const() == Some(1) {
        Ok(rhs)
    } else if rhs.only_const() == Some(1) {
        Ok(lhs)
    } else {
        let req = Relation::eq(&lhs, &rhs);
        if let Some(valid) = req.trivial() {
            if !valid {
                bail!("Cannot match axis lengths {} and {}", lhs, rhs);
            }
        } else {
            reqs.push(req.into());
        }
        Ok(if lhs.complexity() < rhs.complexity() {
            lhs
        } else {
            rhs
        })
    }
}

// -- Monadic Pervasive Functions --
// TODO: Turn these into macros?

pub fn not(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot not character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.not(ctx.uiua)?);
    }
    Ok(dep_info)
}

pub fn sign(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        dep_info.typ = 0;
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.sign(ctx.uiua)?);
    }
    Ok(dep_info)
}

pub fn neg(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.neg(ctx.uiua)?);
    }
    Ok(dep_info)
}

pub fn reciprocal(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the reciprocal of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.recip(ctx.uiua)?);
    }
    Ok(dep_info)
}

pub fn abs(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.abs(ctx.uiua)?);
    }
    Ok(dep_info)
}

pub fn sqrt(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot take the square root of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.sqrt(ctx.uiua)?);
    }
    Ok(dep_info)
}

pub fn exp(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot take the exponential of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.exp(ctx.uiua)?);
    }
    Ok(dep_info)
}

pub fn sin(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the sine of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.sin(ctx.uiua)?);
    }
    Ok(dep_info)
}

pub fn floor(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the floor of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.floor(ctx.uiua)?);
    }
    Ok(dep_info)
}

pub fn ceil(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the ceiling of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.ceil(ctx.uiua)?);
    }
    Ok(dep_info)
}

pub fn round(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the rounded value of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.round(ctx.uiua)?);
    }
    Ok(dep_info)
}

// -- Dyadic Pervasive Functions --

fn demote_known(lhs: &mut ShapeInfo, rhs: &mut ShapeInfo) {
    if let Known(lval) = lhs
        && !matches!(rhs, Known(_))
    {
        let shape = lval.shape.iter().map(Axis::from).collect();
        *lhs = Ranked(shape);
    } else if let Known(rval) = &rhs
        && !matches!(lhs, Known(_))
    {
        let shape = rval.shape.iter().map(Axis::from).collect();
        *rhs = Ranked(shape);
    }
}

fn dyadic_pervasive(
    mut lhs: ShapeInfo,
    mut rhs: ShapeInfo,
    func: fn(Value, Value, &uiua::Uiua) -> uiua::UiuaResult<Value>,
    reqs: &mut Vec<Condition>,
    uiua: &uiua::Uiua,
) -> Result<ShapeInfo> {
    demote_known(&mut lhs, &mut rhs);
    let shape = match (lhs, rhs) {
        (Known(lval), Known(rval)) => {
            // TODO: Don't precompute if there are any fixed axes
            Known(func(lval, rval, uiua)?)
        }
        (Ranked(lshape), Ranked(rshape)) => Ranked(pervade_shapes([lshape, rshape], reqs)?),
        (Ranked(shape), Unranked { prefix, mut suffix })
        | (Unranked { prefix, mut suffix }, Ranked(shape)) => {
            if shape.len() > prefix.len() {
                suffix.clear();
            }
            let prefix = pervade_shapes([shape, prefix], reqs)?;
            Unranked { prefix, suffix }
        }
        (
            Unranked {
                prefix: lprefix,
                suffix: lsuffix,
            },
            Unranked {
                prefix: rprefix,
                suffix: rsuffix,
            },
        ) => {
            todo!()
        }
        _ => unreachable!(),
    };
    Ok(shape)
}

// Dyadic pervasive comparison functions
fn cmp(
    func: fn(Value, Value, &uiua::Uiua) -> uiua::UiuaResult<Value>,
    ineq: bool,
    ctx: AnalyzeCtx,
) -> Result<Info> {
    let [lhs, rhs] = two_args(ctx.dep_infos)?;
    let typ = match (lhs.typ, rhs.typ) {
        (2, 2) => 0,
        (0, 3) | (3, 0) | (3, 3) if ineq => 3,
        (2, _) | (_, 2) => 2,
        _ => 0,
    };

    let shape = dyadic_pervasive(lhs.shape, rhs.shape, func, ctx.reqs, ctx.uiua)?;

    Ok(Info { typ, shape })
}

pub fn eq(ctx: AnalyzeCtx) -> Result<Info> {
    cmp(Value::is_eq, false, ctx)
}

pub fn ne(ctx: AnalyzeCtx) -> Result<Info> {
    cmp(Value::is_ne, false, ctx)
}

pub fn lt(ctx: AnalyzeCtx) -> Result<Info> {
    cmp(Value::other_is_lt, true, ctx)
}

pub fn le(ctx: AnalyzeCtx) -> Result<Info> {
    cmp(Value::other_is_le, true, ctx)
}

pub fn gt(ctx: AnalyzeCtx) -> Result<Info> {
    cmp(Value::other_is_gt, true, ctx)
}

pub fn ge(ctx: AnalyzeCtx) -> Result<Info> {
    cmp(Value::other_is_ge, true, ctx)
}

pub fn add(ctx: AnalyzeCtx) -> Result<Info> {
    let [lhs, rhs] = two_args(ctx.dep_infos)?;
    let typ = match (lhs.typ, rhs.typ) {
        (0, 0) => 0,
        (0, 1) | (1, 0) => 1,
        (2, _) | (_, 2) => 2,
        (0, 3) | (3, 0) | (3, 3) => 3,
        (1, 1) => bail!("Cannot add character and character"),
        (1, 3) => bail!("Cannot add character and complex"),
        (3, 1) => bail!("Cannot add complex and character"),
        (_, 4..) | (4.., _) => unreachable!(),
    };

    let shape = dyadic_pervasive(lhs.shape, rhs.shape, Value::add, ctx.reqs, ctx.uiua)?;

    Ok(Info { typ, shape })
}

pub fn sub(ctx: AnalyzeCtx) -> Result<Info> {
    let [lhs, rhs] = two_args(ctx.dep_infos)?;
    let typ = match (lhs.typ, rhs.typ) {
        (0, 0) | (1, 1) => 0,
        (0, 1) => 1,
        (2, _) | (_, 2) => 2,
        (0, 3) | (3, 0) | (3, 3) => 3,
        (1, 0) => bail!("Cannot subtract character from number"),
        (1, 3) => bail!("Cannot subtract character from complex"),
        (3, 1) => bail!("Cannot subtract complex from character"),
        (_, 4..) | (4.., _) => unreachable!(),
    };

    let shape = dyadic_pervasive(lhs.shape, rhs.shape, Value::sub, ctx.reqs, ctx.uiua)?;

    Ok(Info { typ, shape })
}

pub fn mul(ctx: AnalyzeCtx) -> Result<Info> {
    let [lhs, rhs] = two_args(ctx.dep_infos)?;
    let typ = match (lhs.typ, rhs.typ) {
        (0, 0) => 0,
        (0, 1) | (1, 0) => 1,
        (2, _) | (_, 2) => 2,
        (0, 3) | (3, 0) | (3, 3) => 3,
        (1, 1) => bail!("Cannot multiply character and character"),
        (1, 3) => bail!("Cannot multiply character and complex"),
        (3, 1) => bail!("Cannot multiply complex and character"),
        (_, 4..) | (4.., _) => unreachable!(),
    };

    let shape = dyadic_pervasive(lhs.shape, rhs.shape, Value::mul, ctx.reqs, ctx.uiua)?;

    Ok(Info { typ, shape })
}

pub fn div(ctx: AnalyzeCtx) -> Result<Info> {
    let [lhs, rhs] = two_args(ctx.dep_infos)?;
    let typ = match (lhs.typ, rhs.typ) {
        (0, 0) => 0,
        (0, 1) | (1, 0) => 1,
        (2, _) | (_, 2) => 2,
        (0, 3) | (3, 0) | (3, 3) => 3,
        (1, 1) => bail!("Cannot divide character and character"),
        (1, 3) => bail!("Cannot divide character and complex"),
        (3, 1) => bail!("Cannot divide complex and character"),
        (_, 4..) | (4.., _) => unreachable!(),
    };

    let shape = dyadic_pervasive(lhs.shape, rhs.shape, Value::div, ctx.reqs, ctx.uiua)?;

    Ok(Info { typ, shape })
}

// -- Monadic Array Functions --

pub fn len(ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let shape = match dep_info.shape {
        Known(value) => Known(value.shape.first().copied().unwrap_or(1).into()),
        Ranked(prefix) | Unranked { prefix, .. } => {
            if let Some(len) = prefix.first().and_then(Axis::only_const) {
                if len < 0 {
                    bail!("Inferred negative length of {len}");
                }
                Known((len as usize).into())
            } else {
                Ranked(SymShape::new())
            }
        }
    };
    Ok(Info { typ: 0, shape })
}

pub fn shape(ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let shape = match dep_info.shape {
        Known(value) => Known(value.shape.iter().copied().collect()),
        Ranked(shape) => {
            if let Some(real_shape) = shape
                .iter()
                .map(Axis::only_const)
                .map(|v| v.and_then(|v| (v >= 0).then_some(v as usize)))
                .collect::<Option<Value>>()
            {
                Known(real_shape)
            } else {
                Ranked(smallvec![shape.len().into()])
            }
        }
        Unranked { .. } => Ranked(smallvec![Axis::newvar(ctx.nvars)]),
    };
    Ok(Info { typ: 0, shape })
}

pub fn range(ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ != 0 {
        bail!("Range max should be a single integer or a list of integers");
    }
    // TODO: Add an upper bound to the size of range which will be computed ahead of time?
    let shape = match dep_info.shape {
        Known(value) => Known(value.range(ctx.uiua)?),
        Ranked(mut shape) => {
            if shape.is_empty() {
                Ranked(smallvec![Axis::newvar(ctx.nvars)])
            } else if shape.len() == 1 {
                let len = shape.remove(0);
                if let Some(real_len) = len.only_const() {
                    let mut new_shape = SymShape::new();
                    for _ in 0..real_len {
                        new_shape.push(Axis::newvar(ctx.nvars));
                    }
                    new_shape.push(len);
                    Ranked(new_shape)
                } else {
                    Unranked {
                        prefix: SymShape::new(),
                        suffix: smallvec![len],
                    }
                }
            } else {
                bail!("Range max should be a single integer or a list of integers");
            }
        }
        Unranked {
            mut prefix,
            mut suffix,
        } => {
            let len = if prefix.len() == 1 && suffix.is_empty() {
                prefix.remove(0)
            } else if prefix.is_empty() && suffix.len() == 1 {
                suffix.remove(0)
            } else if prefix.len() == 1 && suffix.len() == 1 {
                let len1 = prefix.remove(0);
                let len2 = suffix.remove(0);
                let req = Relation::eq(&len1, len2);
                if let Some(valid) = req.trivial() {
                    if !valid {
                        bail!("Range max should be a single integer or a list of integers");
                    }
                } else {
                    ctx.reqs.push(req.into());
                }
                len1
            } else {
                bail!("Range max should be a single integer or a list of integers");
            };
            Unranked {
                prefix: SymShape::new(),
                suffix: smallvec![len],
            }
        }
    };

    Ok(Info { typ: 0, shape })
}

pub fn first(ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let shape = match dep_info.shape {
        Known(val) => Known(val.first(ctx.uiua)?),
        Ranked(mut shape) => {
            if shape.is_empty() {
                // Scalar, keep the same shape
                Ranked(shape)
            } else {
                // Remove the first axis, and add a requirement that it be greater than 0
                let len = shape.remove(0);
                let req = Relation::gt(len, 0);
                if let Some(valid) = req.trivial() {
                    if !valid {
                        bail!("Cannot take first of an empty array");
                    }
                } else {
                    ctx.reqs.push(req.into());
                }
                Ranked(shape)
            }
        }
        Unranked { prefix, suffix } => todo!(),
    };

    Ok(Info {
        typ: dep_info.typ,
        shape,
    })
}

pub fn last(ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let shape = match dep_info.shape {
        Known(val) => Known(val.last(ctx.uiua)?),
        Ranked(mut shape) => {
            if shape.is_empty() {
                // Scalar, keep the same shape
                Ranked(shape)
            } else {
                // Remove the first axis, and add a requirement that it be greater than 0
                let len = shape.remove(0);
                let req = Relation::gt(len, 0);
                if let Some(valid) = req.trivial() {
                    if !valid {
                        bail!("Cannot take last of an empty array");
                    }
                } else {
                    ctx.reqs.push(req.into());
                }
                Ranked(shape)
            }
        }
        Unranked { prefix, suffix } => todo!(),
    };

    Ok(Info {
        typ: dep_info.typ,
        shape,
    })
}

pub fn reverse(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        value.reverse();
    }
    Ok(dep_info)
}

pub fn deshape(ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let shape = match dep_info.shape {
        Known(mut value) => {
            value.deshape();
            Known(value)
        }
        Ranked(shape) => {
            let len: Axis = shape.iter().product();
            Ranked(smallvec![len])
        }
        Unranked { .. } => Ranked(smallvec![Axis::newvar(ctx.nvars)]),
    };
    Ok(Info {
        typ: dep_info.typ,
        shape,
    })
}

pub fn deshape_sub(sub: i32, ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let sub_pos = sub.unsigned_abs() as usize;
    let shape = match dep_info.shape {
        // TODO: Needs public method
        Known(mut value) => todo!(),
        Ranked(mut shape) => {
            let rank = shape.len();
            let mut reduce_rank = |n| {
                let reduced = shape.drain(..n + 1).product();
                shape.insert(0, reduced);
            };
            if sub >= 0 {
                if sub_pos <= rank {
                    reduce_rank(rank - sub_pos);
                } else {
                    shape.insert_many(0, std::iter::repeat_n(1.into(), sub_pos - rank));
                }
            } else {
                if sub_pos >= rank {
                    bail!(
                        "Cannot reduce rank {} array by {} ranks",
                        shape.len(),
                        sub_pos
                    );
                }
                reduce_rank(sub_pos);
            }
            Ranked(shape)
        }
        Unranked {
            mut prefix,
            mut suffix,
        } => {
            if sub < 0 && sub_pos < prefix.len() {
                let reduced = prefix.drain(..sub_pos + 1).product();
                prefix.insert(0, reduced);
            } else if sub > 0 {
                prefix.clear();
                if sub_pos < suffix.len() {
                    let reduced = suffix.drain(..suffix.len() - sub_pos + 1).product();
                    suffix.insert(0, reduced);
                }
            }
            Unranked { prefix, suffix }
        }
    };
    Ok(Info {
        typ: dep_info.typ,
        shape,
    })
}

pub fn fix(ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let shape = match dep_info.shape {
        Known(mut value) => {
            value.fix();
            Known(value)
        }
        Ranked(mut shape) => {
            shape.insert(0, 1.into());
            Ranked(shape)
        }
        Unranked { mut prefix, suffix } => {
            prefix.insert(0, 1.into());
            Unranked { prefix, suffix }
        }
    };
    Ok(Info {
        typ: dep_info.typ,
        shape,
    })
}

pub fn bits(ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ != 0 {
        bail!("Argument to bits must be an array of natural numbers");
    }
    let shape = match dep_info.shape {
        Known(value) => Known(value.bits(None, ctx.uiua)?),
        Ranked(mut shape) => {
            shape.push(Axis::newvar(ctx.nvars));
            Ranked(shape)
        }
        Unranked { prefix, mut suffix } => {
            suffix.push(Axis::newvar(ctx.nvars));
            Unranked { prefix, suffix }
        }
    };
    Ok(Info { typ: 0, shape })
}

pub fn transpose(ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let shape = match dep_info.shape {
        Known(mut value) => {
            value.transpose();
            Known(value)
        }
        Ranked(mut shape) => {
            shape.rotate_left(1);
            Ranked(shape)
        }
        Unranked {
            mut prefix,
            mut suffix,
        } => {
            suffix.push(if prefix.is_empty() {
                Axis::newvar(ctx.nvars)
            } else {
                prefix.remove(0)
            });
            Unranked { prefix, suffix }
        }
    };
    Ok(Info {
        typ: dep_info.typ,
        shape,
    })
}

pub fn transpose_n(n: i32, ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

pub fn sort(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        value.sort_up();
    }
    Ok(dep_info)
}

pub fn sort_down(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        value.sort_down();
    }
    Ok(dep_info)
}

pub fn rise(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        *value = value.rise().into();
    }
    Ok(dep_info)
}

pub fn fall(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        *value = value.fall().into();
    }
    Ok(dep_info)
}

pub fn r#where(ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ != 0 {
        bail!("Argument to where must be an array of naturals")
    }
    let shape = match dep_info.shape {
        Known(value) => Known(value.wher(ctx.uiua)?),
        Ranked(shape) => {
            if shape.len() <= 1 {
                Ranked(smallvec![Axis::newvar(ctx.nvars)])
            } else {
                Ranked(smallvec![Axis::newvar(ctx.nvars), shape.len().into()])
            }
        }
        Unranked { .. } => Ranked(smallvec![Axis::newvar(ctx.nvars), Axis::newvar(ctx.nvars)]),
    };
    Ok(Info { typ: 0, shape })
}

pub fn deduplicate(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    match &mut dep_info.shape {
        Known(value) => value.deduplicate(ctx.uiua)?,
        Ranked(prefix) | Unranked { prefix, .. } => {
            if let Some(len) = prefix.first_mut() {
                *len = Axis::newvar(ctx.nvars);
            }
        }
    };
    Ok(dep_info)
}

pub fn classify(ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let shape = match dep_info.shape {
        Known(value) => Known(value.classify()),
        Ranked(mut shape) => {
            if shape.is_empty() {
                Known(0.into())
            } else {
                let len = shape.remove(0);
                Ranked(smallvec![len])
            }
        }
        Unranked { mut prefix, suffix } => {
            if prefix.is_empty() && suffix.is_empty() {
                Unranked { prefix, suffix }
            } else if !prefix.is_empty() {
                let len = prefix.remove(0);
                Ranked(smallvec![len])
            } else {
                Ranked(smallvec![Axis::newvar(ctx.nvars)])
            }
        }
    };
    Ok(Info { typ: 0, shape })
}

pub fn occurrences(mut ctx: AnalyzeCtx) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = dep_info.shape {
        let shape = Known(value.occurrences().into());
        Ok(Info { typ: 0, shape })
    } else {
        ctx.dep_infos = vec![dep_info];
        classify(ctx)
    }
}

pub fn r#box(ctx: AnalyzeCtx) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    Ok(if let Known(value) = &mut dep_info.shape {
        value.box_it();
        dep_info
    } else {
        Info {
            typ: 2,
            shape: Ranked(SymShape::new()),
        }
    })
}

// -- Dyadic Array Functions --

pub fn reshape(ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

pub fn select(ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

pub fn keep(ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

pub fn multi_keep(n: usize, ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

pub fn un_keep(ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

pub fn take(ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

pub fn drop(ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

pub fn couple(ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

pub fn un_couple(ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

pub fn member_of(ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

// -- Misc Functions --

pub fn rand(_ctx: AnalyzeCtx) -> Result<Info> {
    Ok(Info {
        typ: 0,
        shape: Ranked(smallvec![1.into()]),
    })
}

pub fn r#gen(ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

// -- Iterating Modifiers --

pub fn rows(funcs: &[SigNode], ctx: AnalyzeCtx) -> Result<Either<Info, Vec<Info>>> {
    let func = &funcs[0].node;

    // Creates a new variable if any arguments are unranked with an empty prefix
    // Errors if any axis matches fail
    // FIXME: This probably shouldn't short circuit at the first `None`, since it's still possible to find length mismatches even if some of the lengths are unknown
    // FIXME: If the suffix is nonempty, that is enough to know the value is not a scalar, and introduce a new variable for the axis that must match instead of giving up
    let len = ctx
        .dep_infos
        .iter()
        .map(|info| match &info.shape {
            Known(val) => Some(val.shape.first().copied().unwrap_or(1).into()),
            Ranked(shape) => Some(shape.first().cloned().unwrap_or(1.into())),
            Unranked { prefix, .. } => prefix.first().cloned(),
        })
        .try_fold(Ok(1.into()), |acc, next| {
            next.map(|next| match_axes(acc?, next, ctx.reqs))
        })
        .transpose()?
        .unwrap_or_else(|| Axis::newvar(ctx.nvars));

    let mut row_dep_infos = Vec::new();
    for info in &ctx.dep_infos {
        let row_shape = match info.shape.clone() {
            Known(val) => {
                if val.shape.first().copied().unwrap_or(1) == 1 {
                    Known(val.first(ctx.uiua)?)
                } else {
                    Ranked(val.shape[1..].iter().map(Into::into).collect())
                }
            }
            Ranked(mut shape) => {
                if !shape.is_empty() {
                    shape.remove(0);
                }
                Ranked(shape)
            }
            Unranked {
                mut prefix,
                mut suffix,
            } => {
                if !prefix.is_empty() {
                    prefix.remove(0);
                }
                if !suffix.is_empty() {
                    suffix.remove(0);
                }
                Unranked { prefix, suffix }
            }
        };
        row_dep_infos.push(Info {
            typ: info.typ,
            shape: row_shape,
        });
    }

    let process_info = |info: Info| {
        let shape = match info.shape {
            Known(val) => {
                let mut shape: SymShape = val.shape.iter().map(Axis::from).collect();
                shape.insert(0, len.clone());
                Ranked(shape)
            }
            Ranked(mut shape) => {
                shape.insert(0, len.clone());
                Ranked(shape)
            }
            Unranked { mut prefix, suffix } => {
                prefix.insert(0, len.clone());
                Unranked { prefix, suffix }
            }
        };
        Info {
            typ: info.typ,
            shape,
        }
    };

    let data_graph = DataGraph::from_node(func, &ctx.uiua.asm)?;
    let mut info_graph =
        analyze_subgraph(&data_graph, &row_dep_infos, ctx.nvars, ctx.reqs, ctx.uiua)?;
    Ok(if data_graph.stack.len() == 1 {
        let idx = data_graph.stack[0];
        let out_info = info_graph.remove_node(idx).unwrap().1.left().unwrap();
        Either::Left(process_info(out_info))
    } else {
        let mut out_infos = Vec::new();
        for idx in data_graph.stack {
            let out_info = info_graph.remove_node(idx).unwrap().1.left().unwrap();
            out_infos.push(process_info(out_info));
        }
        Either::Right(out_infos)
    })
}

pub fn table(funcs: &[SigNode], ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}

// -- Aggregating Modifiers --

pub fn reduce(funcs: &[SigNode], ctx: AnalyzeCtx) -> Result<Info> {
    todo!()
}
