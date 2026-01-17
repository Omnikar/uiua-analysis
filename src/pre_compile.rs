use itertools::Itertools;
// use anyhow::{bail, Context as _, Result};
// use itertools::Itertools;
use petgraph::{graph::NodeIndex, stable_graph::StableGraph};
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;
use uiua::{Node, Primitive};

use crate::{
    analyze::{axis::Axis, FuncInfos, FuncLib, InfoMap, NodeInfo, RangeInfo, ShapeInfo, ValInfo},
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
    Impl(Impl, usize),
}

/// Custom operations
#[derive(Debug, Clone)]
pub enum Impl {
    Cast(Cast),
    /// Treated analogously to uiua::SysOp::Show
    /// Auto inserted for any values left on the stack at the end of a program
    EndShow,
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
        ValInfo::new(typ, ShapeInfo::Ranked(SmallVec::new()), range)
    }

    pub fn bit_width(&self) -> u8 {
        match self {
            CompType::Int(_, i) => [8, 16, 32, 64][*i as usize],
            CompType::Float(d) => [32, 64][*d as usize],
            CompType::Bool => 1,
            CompType::Char => 32,
        }
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
// impl From<CompType> for RangeInfo {
//     fn from(comp_type: CompType) -> Self {
//         match comp_type {
//             CompType::Int(s, i) => RangeInfo {
//                 extent: 2u64.pow(i + 3),
//                 signed: todo!(),
//                 float: todo!(),
//             },
//             CompType::Float(d) => todo!(),
//             CompType::Bool => todo!(),
//             CompType::Char => todo!(),
//         }
//     }
// }

impl<'u> Op<'u> {
    pub fn span(&self) -> Option<usize> {
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

            (CompType::Int(false, _), CompType::Float(_)) => Some(UtoF),
            (CompType::Int(true, _), CompType::Float(_)) => Some(StoF),

            (CompType::Char, CompType::Int(_, 3)) => Some(UUp),
            (CompType::Char, CompType::Int(_, 0..2)) => Some(UDown),
            (CompType::Char, CompType::Int(_, 2)) => None,
            (CompType::Int(_, 0..2), CompType::Char) => Some(UUp),
            (CompType::Int(_, 3), CompType::Char) => Some(UDown),
            (CompType::Int(_, 2), CompType::Char) => None,
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

    for &(idx, out_i) in &data_graph.stack {
        let new_item =
            prepare_node(idx, &mut pre_compile_graph, &mut annotated_graph, info_map)[out_i];
        pre_compile_graph.stack.push(new_item);
    }

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
        | data @ Data::Node(Node::Prim(Sin, span)) => match_arith_types(data, *span, ctx),
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
    data: Data<'u>,
    span: usize,
    ctx: PreCompileCtx<'u, 'ag, '_, '_, '_, '_, '_>,
) -> &'ag [(NodeIndex, usize)] {
    let out_type = CompType::from_info(&ctx.this_info.vals[0]);
    for ((dep_idx, dep_out_i), dep_info) in ctx.dep_idxs.iter_mut().zip(ctx.dep_infos) {
        let dep_type = CompType::from_info(dep_info);
        // TODO: Handle characters
        if dep_type != out_type
            && let Some(cast) = Cast::from_types(&dep_type, &out_type)
        {
            let cast_op = Op::Impl(Impl::Cast(cast), span);
            let node_info = NodeInfo {
                vals: smallvec![dep_info.clone()],
                subfunc_idxs: vec![],
            };
            let comp_node = CompNode {
                op: cast_op,
                info: node_info,
                types: smallvec![out_type.clone()],
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
