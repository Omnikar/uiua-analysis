pub mod axis;
pub mod impls;

use anyhow::{bail, Context, Result};
use itertools::{Either, Itertools};
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use smallvec::SmallVec;
use uiua::{Node, Uiua, Value};

use crate::graph::{Data, DataGraph, SmallStack};
use axis::{Axis, Condition};

/// Symbolic shape
pub type SymShape = SmallVec<[Axis; 4]>;

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
}

/// Data graph containing static analysis info for each node
#[derive(Clone, Debug)]
pub struct InfoGraph<'a> {
    pub graph: StableGraph<(Data<'a>, Either<Info, Vec<Info>>), usize>,
    /// List of axis relations that must be satisfied for the function to be valid
    pub reqs: Vec<Condition>,
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
    uiua: &Uiua,
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
    reqs: &mut Vec<Condition>,
    idx: NodeIndex,
    arg_infos: &[Info],
    // asm: &Assembly,
    uiua: &Uiua,
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
    use uiua::{ImplPrimitive::*, Primitive::*};
    let info = match data {
        Data::Arg(i) => arg_infos.get(i).context("Insufficient arg info")?.clone(),
        Data::Out => bail!("`Out` node not handled"),
        Data::Node(Node::Push(val)) => Info {
            typ: typ(val),
            shape: ShapeInfo::Known(val.clone()),
        },

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
        _ => todo!(),
    };

    info_graph.node_weight_mut(idx).unwrap().1 = Some(Either::Left(info));

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
