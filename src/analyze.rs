pub mod axis;
pub mod impls;

use anyhow::{bail, Context, Result};
use itertools::{Either, Itertools};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use smallvec::SmallVec;
use uiua::{Assembly, Node};

use crate::graph::{Data, DataGraph, SmallStack};
use axis::{Axis, Relation};

/// Symbolic shape
pub type SymShape = SmallVec<[Axis; 4]>;

/// Statically-inferred shape information about data flowing through a program
#[derive(Clone, Debug)]
pub enum ShapeInfo {
    /// The exact value is known ahead of time
    Known(uiua::Value),
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
}

/// Data graph containing static analysis info for each node
#[derive(Clone, Debug)]
pub struct InfoGraph<'a> {
    pub graph: StableGraph<(Data<'a>, Either<Info, Vec<Info>>), usize>,
    /// List of axis relations that must be satisfied for the function to be valid
    pub reqs: Vec<Relation>,
}

impl ShapeInfo {
    fn rank(&self) -> Option<usize> {
        match self {
            Self::Ranked(shape) => Some(shape.len()),
            _ => None,
        }
    }
}

pub fn analyze_graph<'a>(
    data_graph: &DataGraph<'a>,
    arg_infos: &[Info],
    // asm: &Assembly,
    uiua: &uiua::Uiua,
) -> Result<InfoGraph<'a>> {
    let roots = data_graph.roots(&uiua.asm);

    let mut info_graph = data_graph.graph.map(|_, &data| (data, None), |_, &x| x);

    let mut nvars = 0;
    let mut reqs = Vec::new();

    for root in roots {
        analyze_node(
            &mut info_graph,
            &mut nvars,
            &mut reqs,
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
    let graph = info_graph.map_owned(|_, (data, info)| (data, info.unwrap()), |_, x| x);

    Ok(InfoGraph { graph, reqs })
}

type WorkingInfoGraph<'a> = StableGraph<(Data<'a>, Option<Either<Info, Vec<Info>>>), usize>;

fn analyze_node<'a>(
    info_graph: &mut WorkingInfoGraph<'a>,
    nvars: &mut usize,
    reqs: &mut Vec<Relation>,
    idx: NodeIndex,
    arg_infos: &[Info],
    // asm: &Assembly,
    uiua: &uiua::Uiua,
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
        analyze_node(info_graph, nvars, reqs, dep, arg_infos, uiua)?;
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
        uiua,
    };
    use uiua::ImplPrimitive::*;
    use uiua::Primitive::*;
    let info = match data {
        Data::Arg(i) => arg_infos.get(i).context("Insufficient arg info")?.clone(),
        Data::Out => bail!("`Out` node not handled"),
        Data::Node(Node::Push(val)) => Info {
            typ: typ(val),
            shape: ShapeInfo::Known(val.clone()),
        },

        // -- Monadic Pervasive Functions --
        Data::Node(Node::Prim(Not, span)) => impls::not(ctx, *span)?,
        Data::Node(Node::Prim(Sign, span)) => impls::sign(ctx, *span)?,
        Data::Node(Node::Prim(Neg, span)) => impls::neg(ctx, *span)?,
        Data::Node(Node::Prim(Reciprocal, span)) => impls::reciprocal(ctx, *span)?,
        Data::Node(Node::Prim(Abs, span)) => impls::abs(ctx, *span)?,
        Data::Node(Node::Prim(Sqrt, span)) => impls::sqrt(ctx, *span)?,
        Data::Node(Node::Prim(Exp, span)) => impls::exp(ctx, *span)?,
        Data::Node(Node::Prim(Sin, span)) => impls::sin(ctx, *span)?,
        Data::Node(Node::Prim(Floor, span)) => impls::floor(ctx, *span)?,
        Data::Node(Node::Prim(Ceil, span)) => impls::ceil(ctx, *span)?,
        Data::Node(Node::Prim(Round, span)) => impls::round(ctx, *span)?,

        // -- Monadic Array Functions --
        Data::Node(Node::Prim(Len, span)) => impls::len(ctx, *span)?,
        Data::Node(Node::Prim(Shape, span)) => impls::shape(ctx, *span)?,
        Data::Node(Node::Prim(Range, span)) => impls::range(ctx, *span)?,
        Data::Node(Node::Prim(First, span)) => impls::first(ctx, *span)?,
        Data::Node(Node::Prim(Last, span)) => impls::last(ctx, *span)?,
        Data::Node(Node::Prim(Reverse, span)) => impls::reverse(ctx, *span)?,
        Data::Node(Node::Prim(Deshape, span)) => impls::deshape(ctx, *span)?,
        Data::Node(Node::ImplPrim(DeshapeSub(sub), span)) => impls::deshape_sub(*sub, ctx, *span)?,
        Data::Node(Node::Prim(Fix, span)) => impls::fix(ctx, *span)?,
        Data::Node(Node::Prim(Bits, span)) => impls::bits(ctx, *span)?,
        Data::Node(Node::Prim(Transpose, span)) => impls::transpose(ctx, *span)?,
        Data::Node(Node::ImplPrim(TransposeN(n), span)) => impls::transpose_n(*n, ctx, *span)?,
        Data::Node(Node::Prim(Sort, span)) => impls::sort(ctx, *span)?,
        Data::Node(Node::ImplPrim(SortDown, span)) => impls::sort_down(ctx, *span)?,
        Data::Node(Node::Prim(Rise, span)) => impls::rise(ctx, *span)?,
        Data::Node(Node::Prim(Fall, span)) => impls::fall(ctx, *span)?,
        Data::Node(Node::Prim(Where, span)) => impls::r#where(ctx, *span)?,
        Data::Node(Node::Prim(Deduplicate, span)) => impls::deduplicate(ctx, *span)?,
        Data::Node(Node::Prim(Classify, span)) => impls::classify(ctx, *span)?,
        Data::Node(Node::Prim(Occurrences, span)) => impls::occurrences(ctx, *span)?,
        Data::Node(Node::Prim(Box, span)) => impls::r#box(ctx, *span)?,
        _ => todo!(),
    };

    info_graph.node_weight_mut(idx).unwrap().1 = Some(Either::Left(info));

    Ok(())
}

fn typ(val: &uiua::Value) -> u8 {
    match val {
        uiua::Value::Byte(_) | uiua::Value::Num(_) => 0,
        uiua::Value::Char(_) => 1,
        uiua::Value::Box(_) => 2,
        uiua::Value::Complex(_) => 3,
    }
}
