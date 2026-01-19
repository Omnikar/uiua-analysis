use itertools::Itertools;
// use anyhow::{bail, Context as _, Result};
// use itertools::Itertools;
use petgraph::{
    Direction,
    graph::{EdgeIndex, NodeIndex},
    stable_graph::StableGraph,
    visit::EdgeRef,
};
use smallvec::{SmallVec, smallvec};
use std::collections::HashMap;
use uiua::{Node, Primitive};

use crate::{
    analyze::{FuncInfos, NodeInfo, RangeInfo, ShapeInfo, ValInfo},
    graph::{Data, DataGraph, Stack},
};

type AnnotatedGraph<'u> = StableGraph<(Data<'u>, Option<Vec<(NodeIndex, usize)>>), (usize, usize)>;

#[derive(Debug, Clone)]
pub struct PreCompileGraph<'u> {
    pub graph: StableGraph<CompNode<'u>, (usize, usize)>,
    pub stack: Stack,
    // pub roots: Vec<NodeIndex>,
}

#[derive(Debug, Clone)]
pub struct CompNode<'u> {
    pub op: Op<'u>,
    pub info: NodeInfo,
    pub types: SmallVec<[CompType; 1]>,
}

#[derive(Clone, PartialEq, Eq)]
pub enum CompType {
    /// bool: Signed?
    /// 0-3: 8, 16, 32, or 64 bits?
    Int(bool, u8),
    /// bool: Double precision?
    Float(bool),
    Bool,
    Char,
}

#[derive(Debug, Clone)]
pub enum Op<'u> {
    Data(Data<'u>),
    Prim(Primitive, usize),
    Impl(Impl, usize),
}

/// Custom operations
#[derive(Debug, Clone)]
pub enum Impl {
    Noop,
    Cast(Cast),
    /// Treated analogously to uiua::SysOp::Show
    /// Auto inserted for any values left on the stack at the end of a program
    EndShow,
    Sum,
    Product,
}

#[derive(Debug, Clone, Copy)]
pub enum Cast {
    UUp,
    SUp,
    UDown,
    SDown,
    UtoF,
    StoF,
}

impl<'u> PreCompileGraph<'u> {
    pub fn new() -> Self {
        Self {
            graph: StableGraph::new(),
            stack: SmallVec::new(),
            // roots: Vec::new(),
        }
    }
}

impl CompType {
    pub fn from_info(val_info: &ValInfo) -> Self {
        if val_info.typ == 1 {
            CompType::Char
        } else if val_info.typ != 0 {
            unimplemented!()
        } else if val_info.range.float {
            CompType::Float(true)
        } else if val_info.range == RangeInfo::bool() || val_info.range.extent == 0 {
            CompType::Bool
        } else {
            CompType::Int(
                val_info.range.signed,
                int_type_idx(val_info.range.extent, val_info.range.signed),
            )
        }
    }

    pub fn to_scalar_info(&self) -> ValInfo {
        let (typ, range) = match self {
            CompType::Int(s, i) => (
                0,
                RangeInfo::new(
                    match (s, i) {
                        (false, 0) => u8::MAX as u64,
                        (true, 0) => i8::MAX as u64,
                        (false, 1) => u16::MAX as u64,
                        (true, 1) => i16::MAX as u64,
                        (false, 2) => u32::MAX as u64,
                        (true, 2) => i32::MAX as u64,
                        (false, 3) => u64::MAX,
                        (true, 3) => i64::MAX as u64,
                        (_, 4..) => unreachable!(),
                    },
                    *s,
                    false,
                ),
            ),
            CompType::Float(_) => (0, RangeInfo::new(u64::MAX, true, true)),
            CompType::Bool => (0, RangeInfo::new(1, false, false)),
            CompType::Char => (1, RangeInfo::new(u32::MAX as u64, false, false)),
        };
        ValInfo::new(typ, ShapeInfo::scalar(), range)
    }

    pub fn bit_width(&self) -> u8 {
        match self {
            CompType::Int(_, i) => [8, 16, 32, 64][*i as usize],
            CompType::Float(d) => [32, 64][*d as usize],
            CompType::Bool => 1,
            CompType::Char => 32,
        }
    }

    fn supertype(&self, rhs: &Self) -> Option<Self> {
        use CompType::*;
        Some(match (self, rhs) {
            (Int(s1, i1), Int(s2, i2)) => Int(*s1 || *s2, (*i1).max(*i2)),
            (Int(_, _), Float(d)) | (Float(d), Int(_, _)) => Float(*d),
            (Int(s, i), Bool) | (Bool, Int(s, i)) => Int(*s, *i),
            (Float(d1), Float(d2)) => Float(*d1 || *d2),
            (Float(d), Bool) | (Bool, Float(d)) => Float(*d),
            (Bool, Bool) => Bool,
            (Char, Char) => Char,
            _ => return None,
        })
    }
}
impl std::fmt::Display for CompType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(s, i) => write!(
                f,
                "{}{}",
                if *s { 'i' } else { 'u' },
                ["8", "16", "32", "64"][*i as usize]
            ),
            Self::Float(d) => write!(f, "f{}", if *d { "64" } else { "32" }),
            Self::Bool => write!(f, "bool"),
            Self::Char => write!(f, "char"),
        }
    }
}
impl std::fmt::Debug for CompType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl<'u> Op<'u> {
    pub fn _span(&self) -> Option<usize> {
        match self {
            Op::Data(Data::Node(node)) => node.span(),
            Op::Impl(_, span) => Some(*span),
            _ => None,
        }
    }
}

impl Cast {
    pub fn from_types(from: &CompType, to: &CompType) -> Option<Self> {
        use Cast::*;
        match (from, to) {
            (CompType::Int(false, l), CompType::Int(_, r)) if r > l => Some(UUp),
            (CompType::Int(false, l), CompType::Int(_, r)) if r < l => Some(UDown),
            (CompType::Bool, CompType::Int(_, _)) => Some(UUp),
            (CompType::Int(_, _), CompType::Bool) => Some(UDown),
            (CompType::Int(true, l), CompType::Int(_, r)) if r > l => Some(SUp),
            (CompType::Int(true, l), CompType::Int(_, r)) if r < l => Some(SDown),
            (CompType::Int(_, l), CompType::Int(_, r)) if l == r => None,

            (CompType::Int(false, _) | CompType::Bool, CompType::Float(_)) => Some(UtoF),
            (CompType::Int(true, _), CompType::Float(_)) => Some(StoF),

            (CompType::Char, CompType::Int(_, 3)) => Some(UUp),
            (CompType::Char, CompType::Int(_, 0..2)) => Some(UDown),
            (CompType::Char, CompType::Int(_, 2)) => None,
            (CompType::Int(_, 0..2), CompType::Char) => Some(UUp),
            (CompType::Int(_, 3), CompType::Char) => Some(UDown),
            (CompType::Int(_, 2), CompType::Char) => None,
            (CompType::Float(d1), CompType::Float(d2)) if d1 == d2 => None,
            _ => todo!("Cast from {from} to {to}"),
        }
    }
}
impl From<Cast> for &'static str {
    fn from(cast: Cast) -> Self {
        match cast {
            Cast::UUp => "arith.extui",
            Cast::SUp => "arith.extsi",
            Cast::UDown => "arith.trunci",
            Cast::SDown => "arith.trunci",
            Cast::UtoF => "arith.uitofp",
            Cast::StoF => "arith.sitofp",
        }
    }
}

// TODO: Return a `PreCompileGraph` and a list of roots
pub fn prepare_graph<'u>(
    data_graph: &DataGraph<'u>,
    info_map: &HashMap<NodeIndex, NodeInfo>,
    func_infos: &FuncInfos<'u>,
    uiua: &uiua::Uiua,
) -> PreCompileGraph<'u> {
    let mut annotated_graph: AnnotatedGraph<'u> =
        data_graph.graph.map(|_, &data| (data, None), |_, &e| e);

    let mut pre_compile_graph = PreCompileGraph::new();

    for root in data_graph.roots(&uiua.asm) {
        prepare_node(
            root,
            // 0,
            &mut pre_compile_graph,
            &mut annotated_graph,
            info_map,
        );
    }

    // In order to prevent stored stack references from being invalidated by the graph rewrites, we temporarily store the stack as connections to a single terminal node. This way, the invalidation of stack references is handled by the graph rewrite logic itself.
    let stack_node = CompNode {
        op: Op::Impl(Impl::Noop, usize::MAX),
        info: NodeInfo::no_vals(),
        types: SmallVec::new(),
    };
    let stack_node_idx = pre_compile_graph.graph.add_node(stack_node);

    for (stack_i, &(idx, out_i)) in data_graph.stack.iter().enumerate() {
        let (idx, out_i) =
            prepare_node(idx, &mut pre_compile_graph, &mut annotated_graph, info_map)[out_i];
        pre_compile_graph
            .graph
            .add_edge(stack_node_idx, idx, (out_i, stack_i));
    }

    // - Graph rewrites -
    standardize_cmp(&mut pre_compile_graph, func_infos, uiua);
    reduce_sum_and_product(&mut pre_compile_graph, func_infos, uiua);
    // ---

    pre_compile_graph.stack.extend(std::iter::repeat_n(
        Default::default(),
        data_graph.stack.len(),
    ));
    for edge in pre_compile_graph.graph.edges(stack_node_idx) {
        let (out_i, in_i) = *edge.weight();
        let item = (edge.target(), out_i);
        pre_compile_graph.stack[in_i] = item;
    }

    pre_compile_graph.graph.remove_node(stack_node_idx);

    pre_compile_graph
}

fn prepare_node<'u, 'ag>(
    idx: NodeIndex,
    pre_compile_graph: &mut PreCompileGraph<'u>,
    annotated_graph: &'ag mut AnnotatedGraph<'u>,
    info_map: &HashMap<NodeIndex, NodeInfo>,
) -> &'ag [(NodeIndex, usize)] {
    if annotated_graph.node_weight(idx).unwrap().1.is_some() {
        return annotated_graph
            .node_weight(idx)
            .unwrap()
            .1
            .as_ref()
            .unwrap();
    }

    let (dep_outs, dep_ins): (Vec<_>, Vec<_>) =
        annotated_graph.edges(idx).map(|e| *e.weight()).unzip();
    let mut dep_idxs = annotated_graph
        .neighbors(idx)
        .collect_vec()
        .into_iter()
        .zip(dep_outs)
        .map(|(dep_idx, out_i)| {
            prepare_node(
                dep_idx,
                // *out_i,
                pre_compile_graph,
                annotated_graph,
                info_map,
            )[out_i]
        })
        .collect_vec();
    let dep_infos = dep_idxs
        .iter()
        .map(|&(dep_idx, out_i)| {
            pre_compile_graph
                .graph
                .node_weight(dep_idx)
                .unwrap()
                .info
                .vals[out_i]
                .clone()
        })
        .collect_vec();

    let this_info = info_map.get(&idx).unwrap();

    let ctx = PreCompileCtx {
        idx,
        annotated_graph,
        pre_compile_graph,
        dep_idxs: &mut dep_idxs,
        dep_ins: &dep_ins,
        dep_infos: &dep_infos,
        this_info,
    };

    let this_node = ctx.annotated_graph.node_weight_mut(idx).unwrap();

    use Primitive::*;
    match this_node.0 {
        data @ Data::Node(Node::Prim(Add, span))
        | data @ Data::Node(Node::Prim(Sub, span))
        | data @ Data::Node(Node::Prim(Mul, span))
        | data @ Data::Node(Node::Prim(Div, span))
        | data @ Data::Node(Node::Prim(Not, span))
        | data @ Data::Node(Node::Prim(Neg, span))
        | data @ Data::Node(Node::Prim(Reciprocal, span))
        | data @ Data::Node(Node::Prim(Sqrt, span))
        | data @ Data::Node(Node::Prim(Exp, span))
        | data @ Data::Node(Node::Prim(Sin, span)) => match_arith_types(false, data, *span, ctx),
        data @ Data::Node(Node::Prim(Eq, span))
        | data @ Data::Node(Node::Prim(Ne, span))
        | data @ Data::Node(Node::Prim(Lt, span))
        | data @ Data::Node(Node::Prim(Le, span))
        | data @ Data::Node(Node::Prim(Gt, span))
        | data @ Data::Node(Node::Prim(Ge, span)) => match_arith_types(true, data, *span, ctx),
        data => add_node(data, ctx),
    }
}

struct PreCompileCtx<'u, 'ag, 'pc, 'x, 'i, 'di, 'ti> {
    idx: NodeIndex,
    annotated_graph: &'ag mut AnnotatedGraph<'u>,
    pre_compile_graph: &'pc mut PreCompileGraph<'u>,
    dep_idxs: &'x mut [(NodeIndex, usize)],
    dep_ins: &'i [usize],
    dep_infos: &'di [ValInfo],
    this_info: &'ti NodeInfo,
}

fn add_node<'u, 'ag>(
    data: Data<'u>,
    ctx: PreCompileCtx<'u, 'ag, '_, '_, '_, '_, '_>,
) -> &'ag [(NodeIndex, usize)] {
    let mut comp_types = SmallVec::new();
    for val_info in &ctx.this_info.vals {
        let comp_type = CompType::from_info(val_info);
        comp_types.push(comp_type);
    }
    let comp_node = CompNode {
        op: Op::Data(data),
        info: ctx.this_info.clone(),
        types: comp_types,
    };
    let new_idx = ctx.pre_compile_graph.graph.add_node(comp_node);
    for (&(dep_idx, dep_out_i), &in_i) in ctx.dep_idxs.iter().zip(ctx.dep_ins) {
        ctx.pre_compile_graph
            .graph
            .add_edge(new_idx, dep_idx, (dep_out_i, in_i));
    }

    ctx.annotated_graph
        .node_weight_mut(ctx.idx)
        .unwrap()
        .1
        .insert(
            (0..ctx.this_info.vals.len())
                .map(|out_i| (new_idx, out_i))
                .collect_vec(),
        )
}

fn match_arith_types<'u, 'ag>(
    use_supertype: bool,
    data: Data<'u>,
    span: usize,
    ctx: PreCompileCtx<'u, 'ag, '_, '_, '_, '_, '_>,
) -> &'ag [(NodeIndex, usize)] {
    let target_type = if use_supertype {
        // Target supertype of arguments
        ctx.dep_infos
            .iter()
            .map(CompType::from_info)
            // FIXME: Probably don't use `.expect` here
            .reduce(|a, b| a.supertype(&b).expect("Cannot identify supertype"))
            .expect("Cannot identify supertype of zero arguments")
    } else {
        // Target output type
        CompType::from_info(&ctx.this_info.vals[0])
    };

    for ((dep_idx, dep_out_i), dep_info) in ctx.dep_idxs.iter_mut().zip(ctx.dep_infos) {
        let dep_type = CompType::from_info(dep_info);
        // TODO: Handle characters
        if dep_type != target_type
            && let Some(cast) = Cast::from_types(&dep_type, &target_type)
        {
            let cast_op = Op::Impl(Impl::Cast(cast), span);
            let node_info = NodeInfo {
                vals: smallvec![dep_info.clone()],
                subfunc_idxs: vec![],
            };
            let comp_node = CompNode {
                op: cast_op,
                info: node_info,
                types: smallvec![target_type.clone()],
            };
            let cast_idx = ctx.pre_compile_graph.graph.add_node(comp_node);
            ctx.pre_compile_graph
                .graph
                .add_edge(cast_idx, *dep_idx, (*dep_out_i, 0));
            *dep_idx = cast_idx;
            *dep_out_i = 0;
        }
    }

    add_node(data, ctx)
}

fn int_type_idx(extent: u64, signed: bool) -> u8 {
    let x = extent;
    let s = signed as u32;
    (x.max(2).ilog2() + s).ilog2().saturating_sub(2).min(3) as u8
}

// -- separate file? --

fn standardize_cmp<'u>(
    pre_compile_graph: &mut PreCompileGraph<'u>,
    _func_infos: &FuncInfos<'u>,
    uiua: &uiua::Uiua,
) {
    let mut nodes_to_delete = Vec::<NodeIndex>::new();
    let mut edges_to_delete = Vec::<EdgeIndex>::new();

    for idx in pre_compile_graph.graph.node_indices().collect_vec() {
        let comp_node = pre_compile_graph.graph.node_weight_mut(idx).unwrap();

        match &comp_node.op {
            Op::Data(Data::Node(Node::Prim(Primitive::Ne, span))) => {
                let mut eq_out_info = comp_node.info.vals[0].clone();
                let comp_node_info = comp_node.info.clone();
                let types = comp_node.types.clone();

                let dependents: Vec<(NodeIndex, (usize, usize))> = pre_compile_graph
                    .graph
                    .edges_directed(idx, Direction::Incoming)
                    .map(|edge| (edge.source(), *edge.weight()))
                    .collect_vec();

                let deps: Vec<(NodeIndex, (usize, usize))> = pre_compile_graph
                    .graph
                    .edges_directed(idx, Direction::Outgoing)
                    .map(|edge| (edge.target(), *edge.weight()))
                    .collect_vec();

                if let ShapeInfo::Known(val) = &mut eq_out_info.shape {
                    *val = val.clone().not(uiua).unwrap();
                }

                let eq_comp_node = CompNode {
                    op: Op::Prim(Primitive::Eq, *span),
                    info: NodeInfo::one_val(eq_out_info),
                    types: types.clone(),
                };
                let not_comp_node = CompNode {
                    op: Op::Prim(Primitive::Not, *span),
                    info: comp_node_info,
                    types,
                };

                let eq_node_idx = pre_compile_graph.graph.add_node(eq_comp_node);
                let not_node_idx = pre_compile_graph.graph.add_node(not_comp_node);

                pre_compile_graph
                    .graph
                    .add_edge(not_node_idx, eq_node_idx, (0, 0));

                for (dep_idx, (out_i, in_i)) in deps {
                    pre_compile_graph
                        .graph
                        .add_edge(eq_node_idx, dep_idx, (out_i, in_i));
                }

                for (dependent_idx, (_out_i, in_i)) in dependents {
                    pre_compile_graph
                        .graph
                        .add_edge(dependent_idx, not_node_idx, (0, in_i));
                }

                nodes_to_delete.push(idx);
            }
            Op::Data(Data::Node(Node::Prim(prim @ Primitive::Lt | prim @ Primitive::Le, span))) => {
                let new_prim = match prim {
                    Primitive::Lt => Primitive::Gt,
                    Primitive::Le => Primitive::Ge,
                    _ => unreachable!(),
                };
                comp_node.op = Op::Prim(new_prim, *span);
                let mut edges_to_add = Vec::new();
                for edge in pre_compile_graph.graph.edges(idx) {
                    let (out_i, in_i) = *edge.weight();
                    edges_to_add.push((edge.target(), out_i, 1 - in_i));
                    edges_to_delete.push(edge.id());
                }
                for (target, out_i, in_i) in edges_to_add {
                    pre_compile_graph.graph.add_edge(idx, target, (out_i, in_i));
                }
            }
            _ => {}
        }
    }

    for idx in nodes_to_delete {
        pre_compile_graph.graph.remove_node(idx);
    }
    for edge_idx in edges_to_delete {
        pre_compile_graph.graph.remove_edge(edge_idx);
    }
}

fn reduce_sum_and_product<'u>(
    pre_compile_graph: &mut PreCompileGraph<'u>,
    func_infos: &FuncInfos<'u>,
    uiua: &uiua::Uiua,
) {
    for idx in pre_compile_graph.graph.node_indices().collect_vec() {
        let comp_node = pre_compile_graph.graph.node_weight(idx).unwrap();

        let Op::Data(Data::Node(Node::Mod(Primitive::Reduce, _, _span))) = comp_node.op else {
            continue;
        };

        let sf_idx = comp_node.info.subfunc_idxs[0];
        let (subfunc_graph, _subfunc_info_map) = &func_infos.subfuncs[sf_idx];

        if !subfunc_graph.graph.node_weights().all(|node| match node {
            Data::Node(node) => node.is_pure(&uiua.asm),
            Data::Arg(_) => true,
        }) || subfunc_graph.stack.len() != 1
        {
            continue;
        }

        let (out_idx, _out_i) = subfunc_graph.stack[0];
        let deps = subfunc_graph
            .graph
            .neighbors(out_idx)
            .map(|idx| *subfunc_graph.graph.node_weight(idx).unwrap())
            .collect_vec();

        if deps.len() != 2 || !deps.contains(&Data::Arg(0)) || !deps.contains(&Data::Arg(1)) {
            continue;
        }

        let data = *subfunc_graph.graph.node_weight(out_idx).unwrap();
        let (new_impl, span) = match data {
            Data::Node(Node::Prim(Primitive::Add, span)) => (Impl::Sum, span),
            Data::Node(Node::Prim(Primitive::Mul, span)) => (Impl::Product, span),
            _ => continue,
        };

        let new_comp_node = CompNode {
            op: Op::Impl(new_impl, *span),
            info: comp_node.info.clone(),
            types: comp_node.types.clone(),
        };

        *pre_compile_graph.graph.node_weight_mut(idx).unwrap() = new_comp_node;
    }
}
