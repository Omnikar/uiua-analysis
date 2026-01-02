pub mod axis;
pub mod impls;

use anyhow::{bail, Context, Result};
use itertools::{Either, Itertools};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use smallvec::SmallVec;
use std::collections::HashMap;
use uiua::{Node, Uiua, Value};

use crate::graph::{Data, DataGraph, SmallStack};
use axis::{Axis, Condition};

/// Symbolic shape
pub type SymShape = SmallVec<[Axis; 4]>;

pub struct AnalyzedFunc<'u> {
    pub id: uiua::FunctionId,
    pub graph: DataGraph<'u>,
    pub infos: FuncInfos<'u>,
    pub span: usize,
}

/// Graphs and analysis results for bound functions
pub struct FuncLib<'u> {
    pub funcs: Vec<AnalyzedFunc<'u>>,
}

/// Statically-inferred shape information about data flowing through a program
#[derive(Clone, Debug)]
pub enum ShapeInfo {
    /// The exact value is known ahead of time
    Known(Value),
    /// The rank is known ahead of time
    /// Keeps track of info about axis lengths
    Ranked(SymShape),
    /// The rank is not known ahead of time
    /// Keeps track of some info about axis lengths
    Unranked { prefix: SymShape, suffix: SymShape },
}

/// Statically-inferred information about data flowing through a program
#[derive(Clone, Debug)]
pub struct Info {
    pub typ: u8,
    pub shape: ShapeInfo,
    /// For functions that needed to be analyzed for this node
    pub subfunc_idxs: SmallVec<[usize; 2]>,
}

pub type Infos = HashMap<NodeIndex, Info>;
type WorkingInfoGraph<'u> = StableGraph<(Data<'u>, Option<Either<Info, Vec<Info>>>), usize>;

#[derive(Clone, Debug)]
pub struct FuncInfos<'u> {
    pub map: Infos,
    pub reqs: Vec<Condition>,
    pub subfuncs: Vec<(DataGraph<'u>, Infos)>,
    pub args: SmallVec<[Info; 2]>,
    pub outs: SmallVec<[Info; 1]>,
}

impl<'u> AnalyzedFunc<'u> {
    pub fn new(
        id: uiua::FunctionId,
        graph: DataGraph<'u>,
        infos: FuncInfos<'u>,
        span: usize,
    ) -> Self {
        Self {
            id,
            graph,
            infos,
            span,
        }
    }
}

impl<'u> FuncLib<'u> {
    pub fn new() -> Self {
        Self { funcs: Vec::new() }
    }

    /// Attempts to find a function in the library that matches the function ID and the desired input infos
    pub fn find(
        &self,
        id: &uiua::FunctionId,
        arg_infos: &[Info],
        nvars: &mut usize,
    ) -> Option<(usize, Result<Either<Info, Vec<Info>>>)> {
        self.funcs
            .iter()
            .enumerate()
            .filter_map(|(i, func)| (*id == func.id).then_some((i, &func.infos)))
            .find_map(|(func_i, func_infos)| {
                try_match_func(arg_infos, func_infos, nvars).map(|info| (func_i, info))
            })
    }
}

/// Attempt to match a set of arguments to the an analyzed function
/// If the arguments do not match, returns `None`
/// If the arguments match but do not satisfy the reqs, returns `Some(Err(…))`
/// Otherwise, returns `Some(Ok(…))` with the inferred outputs
fn try_match_func(
    arg_infos: &[Info],
    func_infos: &FuncInfos,
    nvars: &mut usize,
) -> Option<Result<Either<Info, Vec<Info>>>> {
    // dbg!(arg_infos, func_infos);
    // Axis variable substitutions to be made
    let mut substs = HashMap::new();
    for (arg, func_arg) in arg_infos.iter().zip(func_infos.args.iter()) {
        if arg.typ != func_arg.typ {
            return None;
        }

        match (&arg.shape, &func_arg.shape) {
            (ShapeInfo::Known(val), ShapeInfo::Known(func_val)) if *val == *func_val => {}
            (ShapeInfo::Known(val), ShapeInfo::Ranked(func_shape)) => {
                let shape = &val.shape;
                if shape.len() != func_shape.len() {
                    return None;
                }
                for (ax, func_ax) in shape.iter().zip(func_shape.iter()) {
                    if let Some(len) = func_ax.only_const() {
                        if *ax != len as usize {
                            return None;
                        }
                    } else if let Some(var_i) = func_ax.single_var() {
                        substs.insert(var_i, Axis::from(*ax));
                    } else {
                        return None;
                    }
                }
            }
            (ShapeInfo::Ranked(shape), ShapeInfo::Ranked(func_shape)) => {
                if shape.len() != func_shape.len() {
                    return None;
                }
                for (ax, func_ax) in shape.iter().zip(func_shape.iter()) {
                    if let Some(len) = func_ax.only_const() {
                        if ax.only_const()? != len {
                            return None;
                        }
                    } else if let Some(var_i) = func_ax.single_var() {
                        substs.insert(var_i, ax.clone());
                    } else {
                        return None;
                    }
                }
            }
            (
                ShapeInfo::Unranked {
                    prefix: lprefix,
                    suffix: lsuffix,
                },
                ShapeInfo::Unranked {
                    prefix: rprefix,
                    suffix: rsuffix,
                },
            ) => todo!(),
            _ => return None,
        }
    }

    let min_nvars = func_infos
        .reqs
        .iter()
        .map(|req| req.to_nvars())
        .chain(func_infos.outs.iter().map(|info| info.shape.to_nvars()))
        .max()
        .unwrap_or_default();
    for i in 0..min_nvars {
        substs.entry(i).or_insert_with(|| Axis::newvar(nvars));
    }

    Some((|| {
        let reqs = func_infos
            .reqs
            .iter()
            .map(|req| req.substitute(&substs))
            .collect::<Result<Vec<_>>>()?;
        // TODO: More complicated algebra here to make sure the reqs are satisfied?
        for req in &reqs {
            if let Some(false) = req.trivial() {
                // TODO formatting for `Condition`
                bail!("Requirement not satisfied: {req}");
            }
        }

        let mut outs = func_infos
            .outs
            .iter()
            .map(|out_info| {
                out_info
                    .shape
                    .substitute(&substs)
                    .map(|shape| Info::new(out_info.typ, shape))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(match outs.len() {
            1 => Either::Left(outs.pop().unwrap()),
            _ => Either::Right(outs),
        })
    })())
}

impl Info {
    pub fn new(typ: u8, shape: ShapeInfo) -> Self {
        Self {
            typ,
            shape,
            subfunc_idxs: SmallVec::new(),
        }
    }

    pub fn func(mut self, i: usize) -> Self {
        self.subfunc_idxs.push(i);
        self
    }
}

impl ShapeInfo {
    /// Returns the rank if it is known
    fn rank(&self) -> Option<usize> {
        match self {
            Self::Ranked(shape) => Some(shape.len()),
            _ => None,
        }
    }

    /// Returns `Some(Some(length))` if rank ≥1
    /// Returns `Some(None)` if known to be a scalar
    /// Returns `None` if whether it is a scalar is unknown
    fn len(&self) -> Option<Option<Axis>> {
        match self {
            ShapeInfo::Known(val) => Some(val.shape.first().map(Into::into)),
            ShapeInfo::Ranked(shape) => Some(shape.first().cloned()),
            ShapeInfo::Unranked { prefix, .. } => prefix.first().cloned().map(Some),
        }
    }

    fn to_nvars(&self) -> usize {
        match self {
            ShapeInfo::Known(_) => 0,
            ShapeInfo::Ranked(shape) => shape.iter().map(Axis::to_nvars).max().unwrap_or_default(),
            ShapeInfo::Unranked { prefix, suffix } => prefix
                .iter()
                .chain(suffix.iter())
                .map(Axis::to_nvars)
                .max()
                .unwrap_or_default(),
        }
    }

    fn substitute(&self, substs: &HashMap<usize, Axis>) -> Result<Self> {
        match self {
            ShapeInfo::Known(val) => Ok(ShapeInfo::Known(val.clone())),
            ShapeInfo::Ranked(shape) => shape
                .iter()
                .map(|ax| ax.substitute(substs))
                .collect::<Result<_>>()
                .map(ShapeInfo::Ranked),
            ShapeInfo::Unranked { prefix, suffix } => {
                let prefix = prefix
                    .iter()
                    .map(|ax| ax.substitute(substs))
                    .collect::<Result<_>>()?;
                let suffix = suffix
                    .iter()
                    .map(|ax| ax.substitute(substs))
                    .collect::<Result<_>>()?;
                Ok(ShapeInfo::Unranked { prefix, suffix })
            }
        }
    }
}

pub fn analyze_func_graph<'u>(
    data_graph: &DataGraph<'u>,
    arg_infos: &[Info],
    funclib: &mut FuncLib<'u>,
    uiua: &'u Uiua,
) -> Result<FuncInfos<'u>> {
    let mut nvars = arg_infos
        .iter()
        .map(|info| match &info.shape {
            ShapeInfo::Known(_) => 0,
            ShapeInfo::Ranked(shape) => shape.iter().map(Axis::to_nvars).max().unwrap_or(0),
            ShapeInfo::Unranked { prefix, suffix } => prefix
                .iter()
                .chain(suffix.iter())
                .map(Axis::to_nvars)
                .max()
                .unwrap_or(0),
        })
        .max()
        .unwrap_or(0);

    let mut reqs = Vec::new();
    let mut funcs = Vec::new();

    let map = analyze_subgraph(
        data_graph, arg_infos, &mut nvars, &mut reqs, &mut funcs, funclib, uiua,
    )?;

    let outs = data_graph
        .stack
        .iter()
        .map(|idx| map.get(idx).unwrap().clone())
        .collect();

    Ok(FuncInfos {
        map,
        reqs,
        subfuncs: funcs,
        args: arg_infos.into(),
        outs,
    })
}

pub fn analyze_subgraph<'u>(
    data_graph: &DataGraph<'u>,
    arg_infos: &[Info],
    nvars: &mut usize,
    reqs: &mut Vec<Condition>,
    subfuncs: &mut Vec<(DataGraph<'u>, Infos)>,
    funclib: &mut FuncLib<'u>,
    uiua: &'u Uiua,
) -> Result<Infos> {
    let roots = data_graph.roots(&uiua.asm);

    let mut info_graph = data_graph.graph.map(|_, &data| (data, None), |_, &x| x);

    for root in roots {
        analyze_node(
            &mut info_graph,
            nvars,
            reqs,
            subfuncs,
            funclib,
            root,
            arg_infos,
            uiua,
        )?;
    }

    for (_, opt) in info_graph.node_weights() {
        if opt.is_none() {
            bail!("Data graph analysis did not complete");
        }
    }

    // `.unwrap()` should never fail due to the above check

    let indices = info_graph.node_indices().collect_vec();
    let map = indices
        .into_iter()
        .filter_map(|idx| {
            info_graph
                .remove_node(idx)
                .unwrap()
                .1
                .unwrap()
                .left()
                .map(|info| (idx, info))
        })
        .collect();
    Ok(map)
}

#[allow(clippy::too_many_arguments)]
fn analyze_node<'u>(
    info_graph: &mut WorkingInfoGraph<'u>,
    nvars: &mut usize,
    reqs: &mut Vec<Condition>,
    subfuncs: &mut Vec<(DataGraph<'u>, Infos)>,
    funclib: &mut FuncLib<'u>,
    idx: NodeIndex,
    arg_infos: &[Info],
    uiua: &'u Uiua,
) -> Result<()> {
    // Short circuit if this node has been analyzed already
    if info_graph
        .node_weight(idx)
        .context("Index did not exist in graph")?
        .1
        .is_some()
    {
        return Ok(());
    }

    // Dependencies of the current node, i.e. directional neighbors
    let deps = info_graph.neighbors(idx);
    let dep_edges = info_graph.edges(idx);
    // Sort deps using the edge weights so that they are in the correct argument order
    let (deps, dep_edges): (SmallStack, SmallVec<[usize; 4]>) = deps
        .zip(dep_edges.map(|e| *e.weight()))
        .sorted_by_key(|(_, e)| *e)
        .unzip();

    // List of `Info`s corresponding to the dependencies
    let mut dep_infos = Vec::new();
    for &dep in &deps {
        analyze_node(
            info_graph, nvars, reqs, subfuncs, funclib, dep, arg_infos, uiua,
        )?;
        let info = match info_graph
            .node_weight(dep)
            .unwrap()
            .1
            .clone()
            .context("Analysis did not complete")?
        {
            Either::Left(info) => info,
            // Separately handle `Out` nodes which are the only nodes to be connected to multi-output nodes
            Either::Right(infos) => {
                info_graph.node_weight_mut(idx).unwrap().1 =
                    Some(Either::Left(infos[dep_edges[0]].clone()));
                return Ok(());
            }
        };
        dep_infos.push(info);
    }

    // `.unwrap()` should not fail due to check at top of function
    let data = info_graph.node_weight(idx).unwrap().0;

    let ctx = impls::AnalyzeCtx {
        dep_infos,
        nvars,
        reqs,
        subfuncs,
        funclib,
        uiua,
    };
    use uiua::{ImplPrimitive::*, Primitive::*};
    use Either::*;
    let info = match data {
        Data::Arg(i) => Left(arg_infos.get(i).context("Insufficient arg info")?.clone()),
        Data::Out => bail!("`Out` node not handled"),
        Data::Node(Node::Push(val)) => Left(Info::new(typ(val), ShapeInfo::Known(val.clone()))),
        Data::Node(Node::Call(func, span)) => {
            // dbg!(&ctx.dep_infos);
            let mut func_result = ctx.funclib.find(&func.id, &ctx.dep_infos, ctx.nvars);
            if func_result.is_none() {
                let node = &ctx.uiua.asm[func];
                let data_graph = DataGraph::from_node(node, &ctx.uiua.asm)?;
                // Analyze the function on a generic set of arguments at this rank
                // let generic_arg_infos = ctx
                //     .dep_infos
                //     .iter()
                //     .map(|info| {
                //         use ShapeInfo::*;
                //         let mut new_axes = |num: usize| {
                //             std::iter::repeat_n((), num)
                //                 .map(|_| Axis::newvar(ctx.nvars))
                //                 .collect()
                //         };
                //         let shape = match &info.shape {
                //             Known(val) => Ranked(new_axes(val.shape.len())),
                //             Ranked(shape) => Ranked(new_axes(shape.len())),
                //             // TODO: Maybe unranked should just become generic unranked?
                //             Unranked { prefix, suffix } => Unranked {
                //                 prefix: new_axes(prefix.len()),
                //                 suffix: new_axes(suffix.len()),
                //             },
                //         };
                //         Info::new(info.typ, shape)
                //     })
                //     .collect_vec();
                let generic_arg_infos = &ctx.dep_infos;
                let func_infos =
                    analyze_func_graph(&data_graph, &generic_arg_infos, ctx.funclib, ctx.uiua)?;
                ctx.funclib.funcs.push(AnalyzedFunc::new(
                    func.id.clone(),
                    data_graph,
                    func_infos,
                    *span,
                ));

                func_result = ctx.funclib.find(&func.id, &ctx.dep_infos, ctx.nvars);
            }

            let (func_i, out_result) = func_result.context("Failed to analyze called function")?;
            let mut out_info = out_result?;
            match &mut out_info {
                Left(info) => info.subfunc_idxs.push(func_i),
                Right(infos) => {
                    for info in infos {
                        info.subfunc_idxs.push(func_i);
                    }
                }
            }
            out_info
        }

        // -- Monadic Pervasive Functions --
        Data::Node(Node::Prim(Not, _span)) => Left(impls::not(ctx)?),
        Data::Node(Node::Prim(Sign, _span)) => Left(impls::sign(ctx)?),
        Data::Node(Node::Prim(Neg, _span)) => Left(impls::neg(ctx)?),
        Data::Node(Node::Prim(Reciprocal, _span)) => Left(impls::reciprocal(ctx)?),
        Data::Node(Node::Prim(Abs, _span)) => Left(impls::abs(ctx)?),
        Data::Node(Node::Prim(Sqrt, _span)) => Left(impls::sqrt(ctx)?),
        Data::Node(Node::Prim(Exp, _span)) => Left(impls::exp(ctx)?),
        Data::Node(Node::Prim(Sin, _span)) => Left(impls::sin(ctx)?),
        Data::Node(Node::Prim(Floor, _span)) => Left(impls::floor(ctx)?),
        Data::Node(Node::Prim(Ceil, _span)) => Left(impls::ceil(ctx)?),
        Data::Node(Node::Prim(Round, _span)) => Left(impls::round(ctx)?),

        // -- Dyadic Pervasive Functions --
        Data::Node(Node::Prim(Eq, _span)) => Left(impls::eq(ctx)?),
        Data::Node(Node::Prim(Ne, _span)) => Left(impls::ne(ctx)?),
        Data::Node(Node::Prim(Lt, _span)) => Left(impls::lt(ctx)?),
        Data::Node(Node::Prim(Le, _span)) => Left(impls::le(ctx)?),
        Data::Node(Node::Prim(Gt, _span)) => Left(impls::gt(ctx)?),
        Data::Node(Node::Prim(Ge, _span)) => Left(impls::ge(ctx)?),
        Data::Node(Node::Prim(Add, _span)) => Left(impls::add(ctx)?),
        Data::Node(Node::Prim(Sub, _span)) => Left(impls::sub(ctx)?),
        Data::Node(Node::Prim(Mul, _span)) => Left(impls::mul(ctx)?),
        Data::Node(Node::Prim(Div, _span)) => Left(impls::div(ctx)?),

        // -- Monadic Array Functions --
        Data::Node(Node::Prim(Len, _span)) => Left(impls::len(ctx)?),
        Data::Node(Node::Prim(Shape, _span)) => Left(impls::shape(ctx)?),
        Data::Node(Node::Prim(Range, _span)) => Left(impls::range(ctx)?),
        Data::Node(Node::Prim(First, _span)) => Left(impls::first(ctx)?),
        Data::Node(Node::Prim(Last, _span)) => Left(impls::last(ctx)?),
        Data::Node(Node::Prim(Reverse, _span)) => Left(impls::reverse(ctx)?),
        Data::Node(Node::Prim(Deshape, _span)) => Left(impls::deshape(ctx)?),
        Data::Node(Node::ImplPrim(DeshapeSub(sub), _span)) => Left(impls::deshape_sub(*sub, ctx)?),
        Data::Node(Node::Prim(Fix, _span)) => Left(impls::fix(ctx)?),
        Data::Node(Node::Prim(Bits, _span)) => Left(impls::bits(ctx)?),
        Data::Node(Node::Prim(Transpose, _span)) => Left(impls::transpose(ctx)?),
        Data::Node(Node::ImplPrim(TransposeN(n), _span)) => Left(impls::transpose_n(*n, ctx)?),
        Data::Node(Node::Prim(Sort, _span)) => Left(impls::sort(ctx)?),
        Data::Node(Node::ImplPrim(SortDown, _span)) => Left(impls::sort_down(ctx)?),
        Data::Node(Node::Prim(Rise, _span)) => Left(impls::rise(ctx)?),
        Data::Node(Node::Prim(Fall, _span)) => Left(impls::fall(ctx)?),
        Data::Node(Node::Prim(Where, _span)) => Left(impls::r#where(ctx)?),
        Data::Node(Node::Prim(Deduplicate, _span)) => Left(impls::deduplicate(ctx)?),
        Data::Node(Node::Prim(Classify, _span)) => Left(impls::classify(ctx)?),
        Data::Node(Node::Prim(Occurrences, _span)) => Left(impls::occurrences(ctx)?),
        Data::Node(Node::Prim(Box, _span)) => Left(impls::r#box(ctx)?),

        // -- Dyadic Array Functions --

        // -- Misc Functions --
        Data::Node(Node::Prim(Rand, _span)) => Left(impls::rand(ctx)?),

        // -- _________ Modifiers --
        Data::Node(Node::Mod(Rows, funcs, _span)) => impls::rows(funcs, ctx)?,
        Data::Node(Node::Mod(Table, funcs, _span)) => impls::table(funcs, ctx)?,

        // -- Iterating Modifiers --
        Data::Node(Node::Mod(Reduce, funcs, _span)) => Left(impls::reduce(funcs, ctx)?),

        _ => todo!(),
    };

    info_graph.node_weight_mut(idx).unwrap().1 = Some(info);

    Ok(())
}

fn typ(val: &Value) -> u8 {
    match val {
        Value::Byte(_) | Value::Num(_) => 0,
        Value::Char(_) => 1,
        Value::Box(_) => 2,
        Value::Complex(_) => 3,
    }
}

fn typ_name(id: u8) -> &'static str {
    match id {
        0 => "number",
        1 => "character",
        2 => "box",
        3 => "complex",
        n @ 4.. => panic!("Nonexistent type id: {n}"),
    }
}
