//! Primitive-specific functions for propagating static analysis `Info`

use anyhow::{Context, Result, bail};
use itertools::Itertools;
use smallvec::{SmallVec, smallvec};
use uiua::{SigNode, Value};

use super::axis::{Axis, Condition, Relation};
use super::{
    FuncLib, InfoMap, NodeInfo, RangeInfo, ShapeInfo, SymShape, ValInfo, analyze_subgraph, typ_name,
};
use crate::graph::DataGraph;

use ShapeInfo::*;

pub struct AnalyzeCtx<'n, 'r, 'f, 'l, 'u> {
    pub dep_infos: Vec<ValInfo>,
    pub nvars: &'n mut usize,
    pub reqs: &'r mut Vec<Condition>,
    pub subfuncs: &'f mut Vec<(DataGraph<'u>, InfoMap)>,
    pub funclib: &'l mut FuncLib<'u>,
    pub uiua: &'u uiua::Uiua,
}

fn n_args<const N: usize>(dep_infos: Vec<ValInfo>) -> Result<[ValInfo; N]> {
    dep_infos.try_into().ok().context("Incorrect arg count")
}
fn one_arg(mut dep_infos: Vec<ValInfo>) -> Result<ValInfo> {
    dep_infos.pop().context("Incorrect arg count")
}
fn two_args(dep_infos: Vec<ValInfo>) -> Result<[ValInfo; 2]> {
    n_args::<2>(dep_infos)
}

fn known(value: Value) -> (ShapeInfo, RangeInfo) {
    let range = RangeInfo::from_value(&value);
    (ShapeInfo::Known(value), range)
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
                bail!("Cannot match axis lengths {lhs} and {rhs}");
            }
        } else {
            reqs.push(req.into());
        }
        Ok(if lhs.complexity() <= rhs.complexity() {
            lhs
        } else {
            rhs
        })
    }
}

// -- Monadic Pervasive Functions --
// TODO: Turn these into macros?

pub fn not(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot not character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.not(ctx.uiua)?);
    }
    dep_info.range.signed |= dep_info.range.extent > 1;
    Ok(NodeInfo::one_val(dep_info))
}

pub fn sign(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        dep_info.typ = 0;
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.sign(ctx.uiua)?);
    }
    dep_info.range.extent = 1;
    dep_info.range.float = false;
    Ok(NodeInfo::one_val(dep_info))
}

pub fn neg(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.neg(ctx.uiua)?);
    }
    dep_info.range.signed = true;
    Ok(NodeInfo::one_val(dep_info))
}

pub fn reciprocal(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the reciprocal of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.recip(ctx.uiua)?);
    }
    dep_info.range.float = true;
    Ok(NodeInfo::one_val(dep_info))
}

pub fn abs(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.abs(ctx.uiua)?);
    }
    // TODO: Handle this properly
    // dep_info.range.signed = false;
    Ok(NodeInfo::one_val(dep_info))
}

pub fn sqrt(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot take the square root of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.sqrt(ctx.uiua)?);
    }
    dep_info.range.float = true;
    Ok(NodeInfo::one_val(dep_info))
}

pub fn exp(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot take the exponential of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.exp(ctx.uiua)?);
    }
    dep_info.range.float = true;
    Ok(NodeInfo::one_val(dep_info))
}

pub fn sin(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the sine of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.sin(ctx.uiua)?);
    }
    dep_info.range.float = true;
    Ok(NodeInfo::one_val(dep_info))
}

pub fn floor(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the floor of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.floor(ctx.uiua)?);
    }
    dep_info.range.float = false;
    Ok(NodeInfo::one_val(dep_info))
}

pub fn ceil(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the ceiling of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.ceil(ctx.uiua)?);
    }
    dep_info.range.float = false;
    Ok(NodeInfo::one_val(dep_info))
}

pub fn round(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the rounded value of character");
    }
    if let Known(val) = dep_info.shape {
        dep_info.shape = Known(val.round(ctx.uiua)?);
    }
    dep_info.range.float = false;
    Ok(NodeInfo::one_val(dep_info))
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
                prefix: _lprefix,
                suffix: _lsuffix,
            },
            Unranked {
                prefix: _rprefix,
                suffix: _rsuffix,
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
) -> Result<NodeInfo> {
    let [lhs, rhs] = two_args(ctx.dep_infos)?;
    let typ = match (lhs.typ, rhs.typ) {
        (2, 2) => 0,
        (0, 3) | (3, 0) | (3, 3) if ineq => 3,
        (2, _) | (_, 2) => 2,
        _ => 0,
    };

    let shape = dyadic_pervasive(lhs.shape, rhs.shape, func, ctx.reqs, ctx.uiua)?;

    Ok(NodeInfo::one_val(ValInfo::new(
        typ,
        shape,
        RangeInfo::bool(),
    )))
}

pub fn eq(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    cmp(Value::is_eq, false, ctx)
}

pub fn ne(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    cmp(Value::is_ne, false, ctx)
}

pub fn lt(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    cmp(Value::other_is_lt, true, ctx)
}

pub fn le(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    cmp(Value::other_is_le, true, ctx)
}

pub fn gt(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    cmp(Value::other_is_gt, true, ctx)
}

pub fn ge(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    cmp(Value::other_is_ge, true, ctx)
}

pub fn add(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let [lhs, rhs] = two_args(ctx.dep_infos)?;
    let typ = match (lhs.typ, rhs.typ) {
        (0, 0) => 0,
        (0, 1) | (1, 0) => 1,
        (2, _) | (_, 2) => 2,
        (0, 3) | (3, 0) | (3, 3) => 3,
        (1, 1) | (1, 3) | (3, 1) => {
            bail!("Cannot add {} and {}", typ_name(lhs.typ), typ_name(rhs.typ))
        }

        (_, 4..) | (4.., _) => unreachable!(),
    };

    let shape = dyadic_pervasive(lhs.shape, rhs.shape, Value::add, ctx.reqs, ctx.uiua)?;

    let range = lhs.range + rhs.range;

    Ok(NodeInfo::one_val(ValInfo::new(typ, shape, range)))
}

pub fn sub(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let [lhs, rhs] = two_args(ctx.dep_infos)?;
    let typ = match (lhs.typ, rhs.typ) {
        (0, 0) | (1, 1) => 0,
        (0, 1) => 1,
        (2, _) | (_, 2) => 2,
        (0, 3) | (3, 0) | (3, 3) => 3,
        (1, 0) | (1, 3) | (3, 1) => bail!(
            "Cannot subtract {} from {}",
            typ_name(lhs.typ),
            typ_name(rhs.typ),
        ),
        (_, 4..) | (4.., _) => unreachable!(),
    };

    let shape = dyadic_pervasive(lhs.shape, rhs.shape, Value::sub, ctx.reqs, ctx.uiua)?;

    let range = rhs.range - lhs.range;

    Ok(NodeInfo::one_val(ValInfo::new(typ, shape, range)))
}

pub fn mul(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let [lhs, rhs] = two_args(ctx.dep_infos)?;
    let typ = match (lhs.typ, rhs.typ) {
        (0, 0) => 0,
        (0, 1) | (1, 0) => 1,
        (2, _) | (_, 2) => 2,
        (0, 3) | (3, 0) | (3, 3) => 3,
        (1, 1) | (1, 3) | (3, 1) => bail!(
            "Cannot multiply {} and {}",
            typ_name(lhs.typ),
            typ_name(rhs.typ),
        ),
        (_, 4..) | (4.., _) => unreachable!(),
    };

    let shape = dyadic_pervasive(lhs.shape, rhs.shape, Value::mul, ctx.reqs, ctx.uiua)?;

    let range = lhs.range * rhs.range;

    Ok(NodeInfo::one_val(ValInfo::new(typ, shape, range)))
}

pub fn div(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let [lhs, rhs] = two_args(ctx.dep_infos)?;
    let typ = match (lhs.typ, rhs.typ) {
        (0, 0) => 0,
        (0, 1) | (1, 0) => 1,
        (2, _) | (_, 2) => 2,
        (0, 3) | (3, 0) | (3, 3) => 3,
        (1, 1) | (1, 3) | (3, 1) => bail!(
            "Cannot divide {} and {}",
            typ_name(lhs.typ),
            typ_name(rhs.typ),
        ),
        (_, 4..) | (4.., _) => unreachable!(),
    };

    let shape = dyadic_pervasive(lhs.shape, rhs.shape, Value::div, ctx.reqs, ctx.uiua)?;

    let range = rhs.range / lhs.range;

    Ok(NodeInfo::one_val(ValInfo::new(typ, shape, range)))
}

// -- Monadic Array Functions --

pub fn len(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let (shape, range) = match dep_info.shape {
        Known(value) => known(value.shape.first().copied().unwrap_or(1).into()),
        Ranked(prefix) | Unranked { prefix, .. } => {
            if let Some(len) = prefix.first().and_then(Axis::only_const) {
                if len < 0 {
                    bail!("Inferred negative length of {len}");
                }
                known((len as usize).into())
            } else {
                (Ranked(SymShape::new()), RangeInfo::index())
            }
        }
    };
    Ok(NodeInfo::one_val(ValInfo::new(0, shape, range)))
}

pub fn shape(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let (shape, range) = match dep_info.shape {
        Known(value) => known(value.shape.iter().copied().collect()),
        Ranked(shape) => {
            if let Some(real_shape) = shape
                .iter()
                .map(Axis::only_const)
                .map(|v| v.and_then(|v| (v >= 0).then_some(v as usize)))
                .collect::<Option<Value>>()
            {
                known(real_shape)
            } else {
                (Ranked(smallvec![shape.len().into()]), RangeInfo::index())
            }
        }
        Unranked { .. } => (
            Ranked(smallvec![Axis::newvar(ctx.nvars)]),
            RangeInfo::index(),
        ),
    };
    Ok(NodeInfo::one_val(ValInfo::new(0, shape, range)))
}

pub fn range(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ != 0 {
        bail!(
            "Range max should be a single integer or a list of integers, but it is {}",
            typ_name(dep_info.typ),
        );
    }
    let (shape, range) = match dep_info.shape {
        Known(value) => {
            let Some(arr) = value
                .as_num_array()
                .cloned()
                .or_else(|| value.as_byte_array().cloned().map(|v| v.convert()))
            else {
                bail!("Cannot take range of {value}")
            };
            let num_elems = arr.elements().product::<f64>() as usize * arr.element_count();
            // Don't pre-evaluate ranges that would include more than 10k elements
            if num_elems < 10_000 {
                known(value.range(ctx.uiua)?)
            } else {
                let shape = arr
                    .elements()
                    .map(|&n| n as usize)
                    .chain((arr.rank() > 0).then_some(arr.element_count()))
                    .map(Axis::from)
                    .collect();
                let extent = arr.elements().copied().map(|v| v as usize).max();
                let range = RangeInfo::try_index(extent);
                (Ranked(shape), range)
            }
        }
        Ranked(mut shape) => {
            let shape = if shape.is_empty() {
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
                bail!(
                    "Range max should be a single integer or a list of integers, but its rank is {}",
                    shape.len()
                );
            };
            (shape, RangeInfo::nat().signed(dep_info.range.signed))
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
            (
                Unranked {
                    prefix: SymShape::new(),
                    suffix: smallvec![len],
                },
                RangeInfo::nat().signed(dep_info.range.signed),
            )
        }
    };

    Ok(NodeInfo::one_val(ValInfo::new(0, shape, range)))
}

pub fn first(ctx: AnalyzeCtx) -> Result<NodeInfo> {
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
        Unranked { .. } => todo!(),
    };

    Ok(NodeInfo::one_val(ValInfo::new(
        dep_info.typ,
        shape,
        dep_info.range,
    )))
}

pub fn last(ctx: AnalyzeCtx) -> Result<NodeInfo> {
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
        Unranked { .. } => todo!(),
    };

    Ok(NodeInfo::one_val(ValInfo::new(
        dep_info.typ,
        shape,
        dep_info.range,
    )))
}

pub fn reverse(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        value.reverse();
    }
    Ok(NodeInfo::one_val(dep_info))
}

pub fn deshape(ctx: AnalyzeCtx) -> Result<NodeInfo> {
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
    Ok(NodeInfo::one_val(ValInfo::new(
        dep_info.typ,
        shape,
        dep_info.range,
    )))
}

pub fn deshape_sub(sub: i32, ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let sub_pos = sub.unsigned_abs() as usize;
    let shape = match dep_info.shape {
        // TODO: Needs public method
        Known(mut _value) => todo!(),
        Ranked(mut shape) => {
            let rank = shape.len();
            let mut reduce_rank = |n| {
                let reduced = shape.drain(..=n).product();
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
                let reduced = prefix.drain(..=sub_pos).product();
                prefix.insert(0, reduced);
            } else if sub > 0 {
                prefix.clear();
                if sub_pos < suffix.len() {
                    let reduced = suffix.drain(..=suffix.len() - sub_pos).product();
                    suffix.insert(0, reduced);
                }
            }
            Unranked { prefix, suffix }
        }
    };
    Ok(NodeInfo::one_val(ValInfo::new(
        dep_info.typ,
        shape,
        dep_info.range,
    )))
}

pub fn fix(ctx: AnalyzeCtx) -> Result<NodeInfo> {
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
    Ok(NodeInfo::one_val(ValInfo::new(
        dep_info.typ,
        shape,
        dep_info.range,
    )))
}

pub fn bits(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ != 0 {
        bail!(
            "Argument to bits must be an array of natural numbers, but it is {}",
            typ_name(dep_info.typ)
        );
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
    let range = RangeInfo::bool().signed(dep_info.range.signed);
    Ok(NodeInfo::one_val(ValInfo::new(0, shape, range)))
}

pub fn transpose(ctx: AnalyzeCtx) -> Result<NodeInfo> {
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
            // TODO: Remove leftmost axis of suffix?
            suffix.push(if prefix.is_empty() {
                Axis::newvar(ctx.nvars)
            } else {
                prefix.remove(0)
            });
            Unranked { prefix, suffix }
        }
    };
    Ok(NodeInfo::one_val(ValInfo::new(
        dep_info.typ,
        shape,
        dep_info.range,
    )))
}

pub fn transpose_n(n: i32, ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let dep_info = one_arg(ctx.dep_infos)?;

    let shape = match dep_info.shape {
        Known(val) => {
            // TODO: change to known value once it's public
            let mut shape: SymShape = val.shape.iter().map(Axis::from).collect();
            let rot = n.unsigned_abs() as usize;
            if n > 0 {
                shape.rotate_left(rot);
            } else if n < 0 {
                shape.rotate_right(rot);
            }
            Ranked(shape)
        }
        Ranked(mut shape) => {
            let rot = n.unsigned_abs() as usize;
            if n > 0 {
                shape.rotate_left(rot);
            } else if n < 0 {
                shape.rotate_right(rot);
            }
            Ranked(shape)
        }
        Unranked { .. } => todo!(),
    };

    Ok(NodeInfo::one_val(ValInfo::new(
        dep_info.typ,
        shape,
        dep_info.range,
    )))
}

pub fn sort(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        value.sort_up();
    }
    Ok(NodeInfo::one_val(dep_info))
}

pub fn sort_down(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        value.sort_down();
    }
    Ok(NodeInfo::one_val(dep_info))
}

pub fn rise(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        *value = value.rise().into();
    }
    dep_info.range = match dep_info.shape.len() {
        // Known length, output must be less
        Some(Some(len)) => RangeInfo::try_index(len.only_const()),
        // Known scalar
        Some(None) => RangeInfo::zero(),
        // Unknown length
        None => RangeInfo::index(),
    };
    Ok(NodeInfo::one_val(dep_info))
}

pub fn fall(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        *value = value.fall().into();
    }
    dep_info.range = match dep_info.shape.len() {
        // Known length, output must be less
        Some(Some(len)) => RangeInfo::try_index(len.only_const()),
        // Known scalar
        Some(None) => RangeInfo::zero(),
        // Unknown length
        None => RangeInfo::index(),
    };
    Ok(NodeInfo::one_val(dep_info))
}

pub fn r#where(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ != 0 {
        bail!(
            "Argument to where must be an array of naturals, but it is {}",
            typ_name(dep_info.typ),
        )
    }
    let (shape, range) = match dep_info.shape {
        Known(value) => known(value.wher(ctx.uiua)?),
        Ranked(shape) => {
            if shape.len() <= 1 {
                let shape_info = Ranked(smallvec![Axis::newvar(ctx.nvars)]);
                (shape_info, RangeInfo::index())
            } else {
                let shape_info = Ranked(smallvec![Axis::newvar(ctx.nvars), shape.len().into()]);
                (shape_info, RangeInfo::index())
            }
        }
        Unranked { .. } => (
            Ranked(smallvec![Axis::newvar(ctx.nvars), Axis::newvar(ctx.nvars)]),
            RangeInfo::index(),
        ),
    };
    Ok(NodeInfo::one_val(ValInfo::new(0, shape, range)))
}

pub fn deduplicate(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    match &mut dep_info.shape {
        Known(value) => value.deduplicate(ctx.uiua)?,
        Ranked(prefix) | Unranked { prefix, .. } => {
            if let Some(len) = prefix.first_mut() {
                *len = Axis::newvar(ctx.nvars);
            }
        }
    }
    Ok(NodeInfo::one_val(dep_info))
}

pub fn classify(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let dep_info = one_arg(ctx.dep_infos)?;
    let (shape, range) = match dep_info.shape {
        Known(value) => known(value.classify()),
        Ranked(mut shape) => {
            if shape.is_empty() {
                (Known(0.into()), RangeInfo::zero())
            } else {
                let len = shape.remove(0);
                let range = RangeInfo::try_index(len.only_const());
                (Ranked(smallvec![len]), range)
            }
        }
        Unranked { mut prefix, suffix } => {
            let shape_info = if prefix.is_empty() && suffix.is_empty() {
                Unranked { prefix, suffix }
            } else if !prefix.is_empty() {
                let len = prefix.remove(0);
                Ranked(smallvec![len])
            } else {
                Ranked(smallvec![Axis::newvar(ctx.nvars)])
            };
            (shape_info, RangeInfo::index())
        }
    };
    Ok(NodeInfo::one_val(ValInfo::new(0, shape, range)))
}

pub fn occurrences(mut ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = dep_info.shape {
        let (shape, range) = known(value.occurrences().into());
        Ok(NodeInfo::one_val(ValInfo::new(0, shape, range)))
    } else {
        ctx.dep_infos = vec![dep_info];
        classify(ctx)
    }
}

pub fn r#box(ctx: AnalyzeCtx) -> Result<NodeInfo> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    Ok(NodeInfo::one_val(
        if let Known(value) = &mut dep_info.shape {
            value.box_it();
            dep_info
        } else {
            ValInfo::new(2, Ranked(SymShape::new()), RangeInfo::zero())
        },
    ))
}

// -- Dyadic Array Functions --

// pub fn reshape(ctx: AnalyzeCtx) -> Result<NodeInfo> {
//     todo!()
// }

// pub fn select(ctx: AnalyzeCtx) -> Result<NodeInfo> {
//     todo!()
// }

// pub fn keep(ctx: AnalyzeCtx) -> Result<NodeInfo> {
//     todo!()
// }

// pub fn multi_keep(n: usize, ctx: AnalyzeCtx) -> Result<NodeInfo> {
//     todo!()
// }

// pub fn un_keep(ctx: AnalyzeCtx) -> Result<NodeInfo> {
//     todo!()
// }

// pub fn take(ctx: AnalyzeCtx) -> Result<NodeInfo> {
//     todo!()
// }

// pub fn drop(ctx: AnalyzeCtx) -> Result<NodeInfo> {
//     todo!()
// }

// pub fn couple(ctx: AnalyzeCtx) -> Result<NodeInfo> {
//     todo!()
// }

// pub fn un_couple(ctx: AnalyzeCtx) -> Result<NodeInfo> {
//     todo!()
// }

// pub fn member_of(ctx: AnalyzeCtx) -> Result<NodeInfo> {
//     todo!()
// }

// -- Misc Functions --

pub fn rand(_ctx: AnalyzeCtx) -> Result<NodeInfo> {
    Ok(NodeInfo::one_val(ValInfo::new(
        0,
        Ranked(SymShape::new()),
        RangeInfo::new(1, false, true),
    )))
}

// pub fn r#gen(ctx: AnalyzeCtx) -> Result<NodeInfo> {
//     todo!()
// }

// -- Mapping Modifiers --

pub fn rows<'u>(funcs: &'u [SigNode], ctx: AnalyzeCtx<'_, '_, '_, '_, 'u>) -> Result<NodeInfo> {
    let func = &funcs[0].node;

    // Creates a new variable if any arguments are unranked with an empty prefix
    // Errors if any axis matches fail
    // FIXME: This probably shouldn't short circuit at the first `None`, since it's still possible to find length mismatches even if some of the lengths are unknown
    // FIXME: If the suffix is nonempty, that is enough to know the value is not a scalar, and introduce a new variable for the axis that must match instead of giving up
    // FIXME: The current method does not account for the case where all inputs are scalars
    let len = ctx
        .dep_infos
        .iter()
        .map(|info| info.shape.len().map(|len| len.unwrap_or_else(|| 1.into())))
        .try_fold(Ok(1.into()), |acc, next| {
            next.map(|next| match_axes(acc?, next, ctx.reqs))
        })
        .transpose()?
        .unwrap_or_else(|| Axis::newvar(ctx.nvars));

    let mut row_infos = Vec::new();
    for info in &ctx.dep_infos {
        let row_shape = match info.shape.clone() {
            Known(val) => {
                if val.shape.first().copied().unwrap_or(1) == 1 {
                    Known(val.first(ctx.uiua)?)
                } else {
                    Ranked(val.shape[1..].iter().map_into().collect())
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
        row_infos.push(ValInfo::new(info.typ, row_shape, info.range));
    }

    let data_graph = DataGraph::from_node(func, &ctx.uiua.asm)?;
    let infos = analyze_subgraph(
        &data_graph,
        &row_infos,
        ctx.nvars,
        ctx.reqs,
        ctx.subfuncs,
        ctx.funclib,
        ctx.uiua,
    )?;

    let process_info = |info: ValInfo| {
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
        ValInfo::new(info.typ, shape, info.range)
    };

    let out = (data_graph
        .stack
        .iter()
        .map(|&(idx, out_i)| process_info(infos.get(&idx).unwrap().vals[out_i].clone())))
    .collect::<NodeInfo>()
    .func(ctx.subfuncs.len());

    ctx.subfuncs.push((data_graph, infos));

    Ok(out)
}

pub fn table<'u>(funcs: &'u [SigNode], ctx: AnalyzeCtx<'_, '_, '_, '_, 'u>) -> Result<NodeInfo> {
    let func = &funcs[0].node;

    let mut ax_iter = ctx
        .dep_infos
        .iter()
        .map(|info| info.shape.len())
        .filter(|len| !matches!(len, Some(None)))
        .map(Option::flatten)
        .peekable();
    // The only leading axes whose positions in the final shape can be known for sure are those that come before the first input of unknown scalar status.
    let leading_axes: Vec<Axis> = ax_iter
        .by_ref()
        .peeking_take_while(Option::is_some)
        .map(Option::unwrap)
        .collect();
    // True if `leading_axes` is known to not contain all of the leading axes produced by `table`. If true, the final output will be `Unranked`.
    let la_incomplete = ax_iter.next().is_some();

    let mut row_infos = Vec::new();

    for info in &ctx.dep_infos {
        // NOTE: This is copy-pasted from the analogous section of `rows` above. Should it be factored out?
        let row_shape = match info.shape.clone() {
            Known(val) => {
                if val.shape.first().copied().unwrap_or(1) == 1 {
                    Known(val.first(ctx.uiua)?)
                } else {
                    Ranked(val.shape[1..].iter().map_into().collect())
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
        row_infos.push(ValInfo::new(info.typ, row_shape, info.range));
    }

    let data_graph = DataGraph::from_node(func, &ctx.uiua.asm)?;
    let infos = analyze_subgraph(
        &data_graph,
        &row_infos,
        ctx.nvars,
        ctx.reqs,
        ctx.subfuncs,
        ctx.funclib,
        ctx.uiua,
    )?;

    let process_info = |info: ValInfo| {
        let shape = match info.shape {
            Known(val) => {
                let mut shape: SymShape = val.shape.iter().map(Axis::from).collect();
                if la_incomplete {
                    Unranked {
                        prefix: leading_axes.iter().cloned().collect(),
                        suffix: shape,
                    }
                } else {
                    shape.insert_many(0, leading_axes.iter().cloned());
                    Ranked(shape)
                }
            }
            Ranked(mut shape) => {
                if la_incomplete {
                    Unranked {
                        prefix: leading_axes.iter().cloned().collect(),
                        suffix: shape,
                    }
                } else {
                    shape.insert_many(0, leading_axes.iter().cloned());
                    Ranked(shape)
                }
            }
            Unranked { mut prefix, suffix } => {
                if la_incomplete {
                    Unranked {
                        prefix: leading_axes.iter().cloned().collect(),
                        suffix,
                    }
                } else {
                    prefix.insert_many(0, leading_axes.iter().cloned());
                    Unranked { prefix, suffix }
                }
            }
        };
        ValInfo::new(info.typ, shape, info.range)
    };

    let out = data_graph
        .stack
        .iter()
        .map(|&(idx, out_i)| process_info(infos.get(&idx).unwrap().vals[out_i].clone()))
        .collect::<NodeInfo>()
        .func(ctx.subfuncs.len());

    ctx.subfuncs.push((data_graph, infos));

    Ok(out)
}

// -- Iterating Modifiers --

// FIXME: This is not right for anything except pervasives currently
pub fn reduce<'u>(funcs: &'u [SigNode], ctx: AnalyzeCtx<'_, '_, '_, '_, 'u>) -> Result<NodeInfo> {
    let func = &funcs[0].node;

    if ctx.dep_infos.len() != 1 {
        // TODO: Higher adicity reduce
        bail!("Higher-adicity reduce is not currently supported");
    }

    let dep_info = one_arg(ctx.dep_infos)?;

    // NOTE: This is copy-pasted from the analogous section of `rows` above. Should it be factored out?
    let row_shape = match dep_info.shape {
        Known(val) => {
            if val.shape.first().copied().unwrap_or(1) == 1 {
                Known(val.first(ctx.uiua)?)
            } else {
                Ranked(val.shape[1..].iter().map_into().collect())
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
    let row_info = ValInfo::new(dep_info.typ, row_shape, dep_info.range);

    // This will only be true upon reduction on a scalar or rank-1 array, in which case reduction does nothing
    if matches!(row_info.shape, Known(_)) {
        // FIXME: Hitting this branch fails to add the function to the subfuncs list. I'm not certain, but I think this is likely to cause problems later.
        return Ok(NodeInfo::one_val(row_info));
    }

    let data_graph = DataGraph::from_node(func, &ctx.uiua.asm)?;
    let infos = analyze_subgraph(
        &data_graph,
        &[row_info.clone(), row_info.clone()],
        ctx.nvars,
        ctx.reqs,
        ctx.subfuncs,
        ctx.funclib,
        ctx.uiua,
    )?;

    if data_graph.stack.len() != 1 {
        bail!(
            "Reduction function must have exactly one output, but it has {}",
            data_graph.stack.len()
        );
    }

    let (idx, out_i) = *data_graph.stack.first().unwrap();
    let out_info = &infos.get(&idx).unwrap().vals[out_i];

    if row_info.typ != out_info.typ {
        bail!(
            "Reduction function input and output types must match, but they are {} and {}",
            typ_name(row_info.typ),
            typ_name(out_info.typ)
        );
    }

    match (&row_info.shape, &out_info.shape) {
        (Ranked(inshape), Ranked(outshape)) => {
            if inshape.len() != outshape.len() {
                bail!(
                    "Reduction function input and output ranks must match, but they are {} and {}",
                    inshape.len(),
                    outshape.len()
                );
            }
            // for (lax, rax) in inshape.iter().zip(outshape.iter()) {
            //     let req = Relation::eq(lax, rax);
            //     if let Some(valid) = req.trivial() {
            //         if !valid {
            //             bail!("Reduction function input and output shapes must match, but they include axes of length {} and {}", lax, rax);
            //         }
            //     } else {
            //         ctx.reqs.push(req.into());
            //     }
            // }
        }
        (Unranked { .. }, _) | (_, Unranked { .. }) => {
            bail!("Unable to determine input and output ranks for reduction function")
        }
        (Known(_), _) | (_, Known(_)) => unreachable!(),
    }

    if matches!(func, uiua::Node::Prim(prim, _) if prim.class().is_pervasive()) {
        ctx.subfuncs.push((data_graph, infos));
        Ok(NodeInfo::one_val(row_info).func(ctx.subfuncs.len() - 1))
    } else {
        bail!("Reduce is currently only supported for pervasive functions");
    }
}

pub fn do_loop<'u>(funcs: &'u [SigNode], ctx: AnalyzeCtx<'_, '_, '_, '_, 'u>) -> Result<NodeInfo> {
    let body = &funcs[0].node;
    let cond = &funcs[1].node;

    let body_sig = body.sig().ok().context("Failed to infer body signature")?;
    let cond_sig = cond
        .sig()
        .ok()
        .context("Failed to infer condition signature")?;

    // FIXME: Only works for net signatures of |n.n+1

    // let mut cond_in = cond_sig.args();
    // let mut cond_out = cond_sig.outputs();
    // let mut body_in = body_sig.args();
    // let mut body_out = body_sig.outputs();

    // if cond_out < body_in + 1 {
    //     let diff = body_in + 1 - cond_out;
    //     cond_in += diff;
    //     cond_out += diff;
    // } else if body_in + 1 < cond_out {
    //     let diff = cond_out - body_in - 1;
    //     body_in += diff;
    //     body_out += diff;
    // }

    // if cond_in != body_out {
    //     bail!("Currently unsupported signature");
    // }

    if cond_sig.outputs() != cond_sig.args() + 1 || body_sig.outputs() != body_sig.args() {
        bail!("Currently unsupported signature");
    }

    // let mut generic_nvars = 0;

    let mut generic_input_infos = Vec::new();
    // let mut generic_reqs = Vec::new();
    // let mut generic_subfuncs = Vec::new();

    for dep in &ctx.dep_infos {
        let rank = dep
            .shape
            .rank()
            .context("Cannot loop with unknown rank array")?;
        let generic_shape = (0..rank)
            // .map(|_| Axis::newvar(&mut generic_nvars))
            .map(|_| Axis::newvar(ctx.nvars))
            .collect();
        let generic_info = ValInfo {
            typ: dep.typ,
            shape: Ranked(generic_shape),
            range: dep.range,
        };
        generic_input_infos.push(generic_info);
    }

    let cond_data_graph = DataGraph::from_node(cond, &ctx.uiua.asm)?;
    // let cond_infos = analyze_subgraph(
    //     &cond_data_graph,
    //     &generic_input_infos,
    //     &mut generic_nvars,
    //     &mut generic_reqs,
    //     &mut generic_subfuncs,
    //     ctx.funclib,
    //     ctx.uiua,
    // )?;

    // let body_data_graph = DataGraph::from_node(body, &ctx.uiua.asm)?;

    // let body_infos = analyze_subgraph(
    //     &body_data_graph,
    //     &…,
    //     ctx.nvars,
    //     ctx.reqs,
    //     ctx.subfuncs,
    //     ctx.funclib,
    //     ctx.uiua,
    // )?;

    // FIXME: Actually do the idempotence calculations

    let cond_infos = analyze_subgraph(
        &cond_data_graph,
        &generic_input_infos,
        ctx.nvars,
        ctx.reqs,
        ctx.subfuncs,
        ctx.funclib,
        ctx.uiua,
    )?;

    let (cond_idx, cond_out_i) = cond_data_graph.stack[0];
    let cond_info = &cond_infos.get(&cond_idx).unwrap().vals[cond_out_i];
    if cond_info.typ != 0
        || (cond_info.range != RangeInfo::bool() && cond_info.range != RangeInfo::zero())
        || cond_info.shape.rank() != Some(0)
    {
        bail!("Expected condition to be a scalar boolean");
    }

    let mut body_input_infos = cond_data_graph
        .stack
        .iter()
        .skip(1)
        .map(|&(idx, out_i)| cond_infos.get(&idx).unwrap().vals[out_i].clone())
        .collect_vec();

    body_input_infos.extend_from_slice(&generic_input_infos[cond_sig.args()..]);

    let body_data_graph = DataGraph::from_node(body, &ctx.uiua.asm)?;
    let body_infos = analyze_subgraph(
        &body_data_graph,
        &body_input_infos,
        ctx.nvars,
        ctx.reqs,
        ctx.subfuncs,
        ctx.funclib,
        ctx.uiua,
    )?;

    let subfuncs_len = ctx.subfuncs.len();

    let node_info = body_data_graph
        .stack
        .iter()
        .map(|&(idx, out_i)| body_infos.get(&idx).unwrap().vals[out_i].clone())
        .collect::<NodeInfo>()
        .func(subfuncs_len)
        .func(subfuncs_len + 1);

    ctx.subfuncs.push((body_data_graph, body_infos));
    ctx.subfuncs.push((cond_data_graph, cond_infos));

    Ok(node_info)
}
