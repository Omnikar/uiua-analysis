pub mod axis;
pub mod impls;

use anyhow::{bail, Context, Result};
use itertools::Itertools;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;
use uiua::{Node, Purity, RealArrayValue, Uiua, Value};

use crate::graph::{Data, DataGraph, Stack};
use axis::{Axis, Condition};

/// Symbolic shape
pub type SymShape = SmallVec<[Axis; 2]>;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeInfo {
    pub extent: u64,
    pub signed: bool,
    pub float: bool,
}

/// Statically-inferred information about data flowing through a program
#[derive(Clone, Debug)]
pub struct ValInfo {
    pub typ: u8,
    pub shape: ShapeInfo,
    pub range: RangeInfo,
}

/// Statically-inferred information about a particular node of a program
#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// Info about each value output by this node
    pub vals: SmallVec<[ValInfo; 1]>,
    /// For functions that needed to be analyzed for this node
    pub subfunc_idxs: Vec<usize>,
}

pub type InfoMap = HashMap<NodeIndex, NodeInfo>;
type WorkingInfoGraph<'u> = StableGraph<(Data<'u>, Option<NodeInfo>), (usize, usize)>;

#[derive(Clone, Debug)]
pub struct FuncInfos<'u> {
    pub map: InfoMap,
    pub reqs: Vec<Condition>,
    pub subfuncs: Vec<(DataGraph<'u>, InfoMap)>,
    pub args: SmallVec<[ValInfo; 2]>,
    pub outs: SmallVec<[ValInfo; 1]>,
    pub purity: Purity,
}

#[derive(Debug, Clone)]
pub struct AnalyzedFunc<'u> {
    pub id: uiua::FunctionId,
    pub graph: DataGraph<'u>,
    pub infos: FuncInfos<'u>,
    pub span: Option<usize>,
}

/// Graphs and analysis results for bound functions
#[derive(Debug, Clone)]
pub struct FuncLib<'u> {
    pub funcs: Vec<AnalyzedFunc<'u>>,
}

impl<'u> AnalyzedFunc<'u> {
    pub fn new(
        id: uiua::FunctionId,
        graph: DataGraph<'u>,
        infos: FuncInfos<'u>,
        span: Option<usize>,
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
        arg_infos: &[ValInfo],
        nvars: &mut usize,
    ) -> Option<(usize, Result<NodeInfo>)> {
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
    arg_infos: &[ValInfo],
    func_infos: &FuncInfos,
    nvars: &mut usize,
) -> Option<Result<NodeInfo>> {
    // dbg!(arg_infos, func_infos);
    // Axis variable substitutions to be made
    let mut substs = HashMap::new();
    for (arg, func_arg) in arg_infos.iter().zip(func_infos.args.iter()) {
        if arg.typ != func_arg.typ {
            return None;
        }

        if crate::pre_compile::CompType::from_info(arg)
            != crate::pre_compile::CompType::from_info(func_arg)
        {
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

        let outs = func_infos.outs.iter().map(|out_info| {
            out_info
                .shape
                .substitute(&substs)
                .map(|shape| ValInfo::new(out_info.typ, shape, out_info.range))
        });

        NodeInfo::try_many_vals(outs)
    })())
}

impl ValInfo {
    pub fn new(typ: u8, shape: ShapeInfo, range: RangeInfo) -> Self {
        Self { typ, shape, range }
    }

    pub fn from_value(value: Value) -> Self {
        let (typ, range) = (typ(&value), RangeInfo::from_value(&value));
        Self::new(typ, ShapeInfo::Known(value), range)
    }
}

impl NodeInfo {
    pub fn one_val(val_info: ValInfo) -> Self {
        Self {
            vals: smallvec![val_info],
            subfunc_idxs: Vec::new(),
        }
    }

    pub fn many_vals(val_infos: impl Iterator<Item = ValInfo>) -> Self {
        Self {
            vals: val_infos.collect(),
            subfunc_idxs: Vec::new(),
        }
    }

    pub fn try_many_vals<E>(
        val_infos: impl Iterator<Item = Result<ValInfo, E>>,
    ) -> Result<Self, E> {
        val_infos.collect::<Result<_, _>>().map(|val_infos| Self {
            vals: val_infos,
            subfunc_idxs: Vec::new(),
        })
    }

    pub fn no_vals() -> Self {
        Self {
            vals: SmallVec::new(),
            subfunc_idxs: Vec::new(),
        }
    }

    pub fn func(mut self, i: usize) -> Self {
        self.subfunc_idxs.push(i);
        self
    }
}
impl FromIterator<ValInfo> for NodeInfo {
    fn from_iter<T: IntoIterator<Item = ValInfo>>(iter: T) -> Self {
        Self::many_vals(iter.into_iter())
    }
}

impl ShapeInfo {
    /// Returns the rank if it is known
    pub fn rank(&self) -> Option<usize> {
        match self {
            Self::Known(val) => Some(val.shape.len()),
            Self::Ranked(shape) => Some(shape.len()),
            _ => None,
        }
    }

    /// Returns `Some(Some(length))` if rank ≥1
    /// Returns `Some(None)` if known to be a scalar
    /// Returns `None` if whether it is a scalar is unknown
    pub fn len(&self) -> Option<Option<Axis>> {
        match self {
            ShapeInfo::Known(val) => Some(val.shape.first().map(Into::into)),
            ShapeInfo::Ranked(shape) => Some(shape.first().cloned()),
            ShapeInfo::Unranked { prefix, .. } => prefix.first().cloned().map(Some),
        }
    }

    pub fn to_nvars(&self) -> usize {
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

    pub fn substitute(&self, substs: &HashMap<usize, Axis>) -> Result<Self> {
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

impl RangeInfo {
    pub fn from_value(value: &Value) -> Self {
        let mut range = Self::new(0, false, false);
        match value {
            Value::Byte(array) => {
                for elem in array.elements() {
                    range.expand(*elem as u64);
                }
            }
            Value::Num(array) => {
                for elem in array.elements() {
                    range.float |= !elem.is_int();
                    range.signed |= *elem < 0.0;
                    range.expand(elem.abs().ceil() as u64);
                }
            }
            Value::Char(array) => {
                for elem in array.elements() {
                    range.expand(*elem as u64);
                }
            }
            Value::Box(_array) => todo!(),
            Value::Complex(_array) => unimplemented!(),
        }
        range
    }

    pub fn new(extent: u64, signed: bool, float: bool) -> Self {
        Self {
            extent,
            signed,
            float,
        }
    }

    pub fn uint(extent: u64) -> Self {
        Self::new(extent, false, false)
    }

    pub fn bool() -> Self {
        Self::uint(1)
    }

    pub fn index() -> Self {
        Self::uint(usize::MAX as u64)
    }

    pub fn try_index<T: TryInto<u64>>(extent: Option<T>) -> Self {
        Self::uint(
            extent
                .and_then(|x| x.try_into().ok())
                .map(|x| x - 1)
                .unwrap_or(usize::MAX as u64),
        )
    }

    pub fn nat() -> Self {
        Self::uint(u64::MAX)
    }

    pub fn zero() -> Self {
        Self::uint(0)
    }

    pub fn expand(&mut self, num: u64) {
        self.extent = self.extent.max(num)
    }

    pub fn signed(mut self, signed: bool) -> Self {
        self.signed = signed;
        self
    }

    pub fn float(mut self, float: bool) -> Self {
        self.float = float;
        self
    }

    pub fn max(mut self, rhs: Self) -> Self {
        self.extent = self.extent.max(rhs.extent);
        self.signed |= rhs.signed;
        self.float |= rhs.float;
        self
    }
}

impl std::ops::Add for RangeInfo {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut range = self.max(rhs);
        range.expand(self.extent.saturating_add(rhs.extent));
        range
    }
}
impl std::ops::Sub for RangeInfo {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut range = self.max(rhs);
        range.expand(self.extent.saturating_add(rhs.extent));
        range.signed(true)
    }
}
impl std::ops::Mul for RangeInfo {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut range = self.max(rhs);
        range.expand(self.extent.saturating_mul(rhs.extent));
        range
    }
}
impl std::ops::Div for RangeInfo {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(mut self, rhs: Self) -> Self {
        self.extent = 2u64.pow(53);
        self.signed |= rhs.signed;
        self.float = true;
        self
    }
}

pub fn analyze_func_graph<'u>(
    data_graph: &DataGraph<'u>,
    arg_infos: &[ValInfo],
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
        .map(|&(idx, out_i)| map.get(&idx).unwrap().vals[out_i].clone())
        .collect();

    let purity = data_graph
        .graph
        .node_weights()
        .map(|data| match data {
            Data::Node(node) => {
                if node.is_pure(&uiua.asm) {
                    Purity::Pure
                } else if node.is_min_purity(Purity::Impure, &uiua.asm) {
                    Purity::Impure
                } else {
                    Purity::Mutating
                }
            }
            _ => Purity::Pure,
        })
        .min()
        .unwrap_or(Purity::Pure);

    Ok(FuncInfos {
        map,
        reqs,
        subfuncs: funcs,
        args: arg_infos.into(),
        outs,
        purity,
    })
}

pub fn analyze_subgraph<'u>(
    data_graph: &DataGraph<'u>,
    arg_infos: &[ValInfo],
    nvars: &mut usize,
    reqs: &mut Vec<Condition>,
    subfuncs: &mut Vec<(DataGraph<'u>, InfoMap)>,
    funclib: &mut FuncLib<'u>,
    uiua: &'u Uiua,
) -> Result<InfoMap> {
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
        .map(|idx| (idx, info_graph.remove_node(idx).unwrap().1.unwrap()))
        .collect();
    Ok(map)
}

#[allow(clippy::too_many_arguments)]
fn analyze_node<'u>(
    info_graph: &mut WorkingInfoGraph<'u>,
    nvars: &mut usize,
    reqs: &mut Vec<Condition>,
    subfuncs: &mut Vec<(DataGraph<'u>, InfoMap)>,
    funclib: &mut FuncLib<'u>,
    idx: NodeIndex,
    arg_infos: &[ValInfo],
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
    let deps: Stack = deps
        .zip(dep_edges.map(|e| *e.weight()))
        .sorted_by_key(|(_, (_, in_i))| *in_i)
        .map(|(idx, (out_i, _))| (idx, out_i))
        .collect();

    // List of `ValInfo`s corresponding to the dependencies
    let mut dep_infos = Vec::new();
    for &(dep, dep_out_i) in &deps {
        analyze_node(
            info_graph, nvars, reqs, subfuncs, funclib, dep, arg_infos, uiua,
        )?;
        let info = info_graph
            .node_weight(dep)
            .unwrap()
            .1
            .as_ref()
            .context("Analysis did not complete")?
            .vals[dep_out_i]
            .clone();
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
    let info = match data {
        Data::Arg(i) => {
            NodeInfo::one_val(arg_infos.get(i).context("Insufficient arg info")?.clone())
        }
        Data::Node(Node::Push(val)) => NodeInfo::one_val(ValInfo::from_value(val.clone())),
        Data::Node(Node::Call(func, span)) => {
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
                let generic_arg_infos = ctx
                    .dep_infos
                    .iter()
                    .map(|info| {
                        if let ShapeInfo::Known(val) = &info.shape {
                            let shape = val.shape.iter().map(Axis::from).collect();
                            let mut info = info.clone();
                            info.shape = ShapeInfo::Ranked(shape);
                            info
                        } else {
                            info.clone()
                        }
                    })
                    .collect_vec();
                let func_infos =
                    analyze_func_graph(&data_graph, &generic_arg_infos, ctx.funclib, ctx.uiua)?;
                ctx.funclib.funcs.push(AnalyzedFunc::new(
                    func.id.clone(),
                    data_graph,
                    func_infos,
                    Some(*span),
                ));

                func_result = ctx.funclib.find(&func.id, &ctx.dep_infos, ctx.nvars);
            }

            let (func_i, out_result) = func_result.context("Failed to analyze called function")?;
            let mut out_info = out_result?;
            out_info.subfunc_idxs.push(func_i);

            out_info
        }

        // -- Monadic Pervasive Functions --
        Data::Node(Node::Prim(Not, _span)) => impls::not(ctx)?,
        Data::Node(Node::Prim(Sign, _span)) => impls::sign(ctx)?,
        Data::Node(Node::Prim(Neg, _span)) => impls::neg(ctx)?,
        Data::Node(Node::Prim(Reciprocal, _span)) => impls::reciprocal(ctx)?,
        Data::Node(Node::Prim(Abs, _span)) => impls::abs(ctx)?,
        Data::Node(Node::Prim(Sqrt, _span)) => impls::sqrt(ctx)?,
        Data::Node(Node::Prim(Exp, _span)) => impls::exp(ctx)?,
        Data::Node(Node::Prim(Sin, _span)) => impls::sin(ctx)?,
        Data::Node(Node::Prim(Floor, _span)) => impls::floor(ctx)?,
        Data::Node(Node::Prim(Ceil, _span)) => impls::ceil(ctx)?,
        Data::Node(Node::Prim(Round, _span)) => impls::round(ctx)?,

        // -- Dyadic Pervasive Functions --
        Data::Node(Node::Prim(Eq, _span)) => impls::eq(ctx)?,
        Data::Node(Node::Prim(Ne, _span)) => impls::ne(ctx)?,
        Data::Node(Node::Prim(Lt, _span)) => impls::lt(ctx)?,
        Data::Node(Node::Prim(Le, _span)) => impls::le(ctx)?,
        Data::Node(Node::Prim(Gt, _span)) => impls::gt(ctx)?,
        Data::Node(Node::Prim(Ge, _span)) => impls::ge(ctx)?,
        Data::Node(Node::Prim(Add, _span)) => impls::add(ctx)?,
        Data::Node(Node::Prim(Sub, _span)) => impls::sub(ctx)?,
        Data::Node(Node::Prim(Mul, _span)) => impls::mul(ctx)?,
        Data::Node(Node::Prim(Div, _span)) => impls::div(ctx)?,

        // -- Monadic Array Functions --
        Data::Node(Node::Prim(Len, _span)) => impls::len(ctx)?,
        Data::Node(Node::Prim(Shape, _span)) => impls::shape(ctx)?,
        Data::Node(Node::Prim(Range, _span)) => impls::range(ctx)?,
        Data::Node(Node::Prim(First, _span)) => impls::first(ctx)?,
        Data::Node(Node::Prim(Last, _span)) => impls::last(ctx)?,
        Data::Node(Node::Prim(Reverse, _span)) => impls::reverse(ctx)?,
        Data::Node(Node::Prim(Deshape, _span)) => impls::deshape(ctx)?,
        Data::Node(Node::ImplPrim(DeshapeSub(sub), _span)) => impls::deshape_sub(*sub, ctx)?,
        Data::Node(Node::Prim(Fix, _span)) => impls::fix(ctx)?,
        Data::Node(Node::Prim(Bits, _span)) => impls::bits(ctx)?,
        Data::Node(Node::Prim(Transpose, _span)) => impls::transpose(ctx)?,
        Data::Node(Node::ImplPrim(TransposeN(n), _span)) => impls::transpose_n(*n, ctx)?,
        Data::Node(Node::Prim(Sort, _span)) => impls::sort(ctx)?,
        Data::Node(Node::ImplPrim(SortDown, _span)) => impls::sort_down(ctx)?,
        Data::Node(Node::Prim(Rise, _span)) => impls::rise(ctx)?,
        Data::Node(Node::Prim(Fall, _span)) => impls::fall(ctx)?,
        Data::Node(Node::Prim(Where, _span)) => impls::r#where(ctx)?,
        Data::Node(Node::Prim(Deduplicate, _span)) => impls::deduplicate(ctx)?,
        Data::Node(Node::Prim(Classify, _span)) => impls::classify(ctx)?,
        Data::Node(Node::Prim(Occurrences, _span)) => impls::occurrences(ctx)?,
        Data::Node(Node::Prim(Box, _span)) => impls::r#box(ctx)?,

        // -- Dyadic Array Functions --

        // -- Misc Functions --
        Data::Node(Node::Prim(Rand, _span)) => impls::rand(ctx)?,

        // -- _________ Modifiers --
        Data::Node(Node::Mod(Rows, funcs, _span)) => impls::rows(funcs, ctx)?,
        Data::Node(Node::Mod(Table, funcs, _span)) => impls::table(funcs, ctx)?,

        // -- Iterating Modifiers --
        Data::Node(Node::Mod(Reduce, funcs, _span)) => impls::reduce(funcs, ctx)?,

        // -- Not yet categorized --
        Data::Node(Node::Prim(Sys(uiua::SysOp::Print), _span)) => NodeInfo::no_vals(),

        _ => todo!("{data:?}"),
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
