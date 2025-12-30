//! Primitive-specific functions for propagating static analysis `Info`

use anyhow::{bail, Context, Result};
use smallvec::smallvec;
use uiua::{Shape, SigNode, Value};

use super::axis::{Axis, Relation};
use super::{Info, ShapeInfo, SymShape};
// use crate::graph::{Data, DataGraph, SmallStack};

use ShapeInfo::*;

pub struct AnalyzeCtx<'a, 'b, 'c> {
    pub dep_infos: Vec<Info>,
    pub nvars: &'a mut usize,
    pub reqs: &'b mut Vec<Relation>,
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

// pub fn pervasive(
//     func: impl Fn(f64, f64) -> f64,
//     dep_infos: Vec<Info>,
//     _nvars: &mut usize,
//     _reqs: &mut Vec<(Axis, Axis)>,
//     _span: usize,
// ) -> Result<Info> {
//     let [mut lhs, mut rhs] = two_args(dep_infos)?;

//     for info in [&mut lhs, &mut rhs] {
//         if let Known(val) = &info.shape {
//             info.shape = Ranked(val.shape.iter().copied().map(Axis::from).collect());
//         }
//     }

//     let shape = match (lhs.shape, rhs.shape) {
//         (Ranked(lshape), Ranked(rshape)) => {
//             todo!()
//         }
//         // (Ranked(lshape), Unranked { prefix, suffix }) => todo!(),
//         // (Unranked { prefix, suffix }, Ranked(small_vec)) => todo!(),
//         // (Unranked { prefix, suffix }, Unranked { prefix, suffix }) => todo!(),
//         _ => todo!(),
//         // _ => unreachable!(),
//     };

//     todo!()
// }

// pub fn pervasive_dyadic(
//     func: impl Fn(Value, Value) -> Value,
//     ctx: AnalyzeCtx,
//     _span: usize,
// ) -> Result<Info> {
//     let [mut lhs, mut rhs] = two_args(ctx.dep_infos)?;

//     todo!()
// }

// -- Monadic Pervasive Functions --
// TODO: Turn these into macros?

pub fn not(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot not character");
    }
    if let Known(val) = &mut dep_info.shape {
        // TODO: `Value` function once public
    }
    Ok(dep_info)
}

pub fn sign(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        dep_info.typ = 0;
    }
    if let Known(val) = &mut dep_info.shape {
        // TODO: `Value` function once public
    }
    Ok(dep_info)
}

pub fn neg(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(val) = &mut dep_info.shape {
        // TODO: `Value` function once public
    }
    Ok(dep_info)
}

pub fn reciprocal(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the reciprocal of character");
    }
    if let Known(val) = &mut dep_info.shape {
        // TODO: `Value` function once public
    }
    Ok(dep_info)
}

pub fn abs(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(val) = &mut dep_info.shape {
        // TODO: `Value` function once public
    }
    Ok(dep_info)
}

pub fn sqrt(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot take the square root of character");
    }
    if let Known(val) = &mut dep_info.shape {
        // TODO: `Value` function once public
    }
    Ok(dep_info)
}

pub fn exp(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot take the exponential of character");
    }
    if let Known(val) = &mut dep_info.shape {
        // TODO: `Value` function once public
    }
    Ok(dep_info)
}

pub fn sin(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the sine of character");
    }
    if let Known(val) = &mut dep_info.shape {
        // TODO: `Value` function once public
    }
    Ok(dep_info)
}

pub fn floor(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the floor of character");
    }
    if let Known(val) = &mut dep_info.shape {
        // TODO: `Value` function once public
    }
    Ok(dep_info)
}

pub fn ceil(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the ceiling of character");
    }
    if let Known(val) = &mut dep_info.shape {
        // TODO: `Value` function once public
    }
    Ok(dep_info)
}

pub fn round(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ == 1 {
        bail!("Cannot get the rounded value of character");
    }
    if let Known(val) = &mut dep_info.shape {
        // TODO: `Value` function once public
    }
    Ok(dep_info)
}

// -- Monadic Array Functions --

pub fn len(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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

pub fn shape(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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

pub fn range(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    if dep_info.typ != 0 {
        bail!("Range max should be a single integer or a list of integers");
    }
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
                    ctx.reqs.push(req);
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

pub fn first(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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
                    ctx.reqs.push(req);
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

pub fn last(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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
                    ctx.reqs.push(req);
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

pub fn reverse(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        value.reverse();
    }
    Ok(dep_info)
}

pub fn deshape(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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

pub fn deshape_sub(sub: i32, ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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

pub fn fix(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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

pub fn bits(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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

pub fn transpose(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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

pub fn transpose_n(n: i32, ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn sort(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        value.sort_up();
    }
    Ok(dep_info)
}

pub fn sort_down(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        value.sort_down();
    }
    Ok(dep_info)
}

pub fn rise(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        *value = value.rise().into();
    }
    Ok(dep_info)
}

pub fn fall(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    let mut dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = &mut dep_info.shape {
        *value = value.fall().into();
    }
    Ok(dep_info)
}

pub fn r#where(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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

pub fn deduplicate(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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

pub fn classify(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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

pub fn occurrences(mut ctx: AnalyzeCtx, span: usize) -> Result<Info> {
    let dep_info = one_arg(ctx.dep_infos)?;
    if let Known(value) = dep_info.shape {
        let shape = Known(value.occurrences().into());
        Ok(Info { typ: 0, shape })
    } else {
        ctx.dep_infos = vec![dep_info];
        classify(ctx, span)
    }
}

pub fn r#box(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
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

pub fn reshape(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn select(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn keep(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn multi_keep(n: usize, ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn un_keep(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn take(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn drop(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn couple(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn un_couple(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn member_of(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

// -- Misc Functions --

pub fn rand(_ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    Ok(Info {
        typ: 0,
        shape: Ranked(smallvec![1.into()]),
    })
}

pub fn r#gen(ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

// -- Iterating Modifiers --

pub fn rows(funcs: &[SigNode], ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn table(funcs: &[SigNode], ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}

pub fn reduce(funcs: &[SigNode], ctx: AnalyzeCtx, _span: usize) -> Result<Info> {
    todo!()
}
