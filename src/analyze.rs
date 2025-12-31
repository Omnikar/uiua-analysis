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
}

pub fn analyze_graph<'a>(
    data_graph: &DataGraph<'a>,
    arg_infos: &[Info],
    uiua: &Uiua,
) -> Result<InfoGraph<'a>> {
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

    let graph = analyze_subgraph(data_graph, arg_infos, &mut nvars, &mut reqs, uiua)?;
    Ok(InfoGraph { graph, reqs })
}

type CompletedInfoGraph<'a> = StableGraph<(Data<'a>, Either<Info, Vec<Info>>), usize>;
type WorkingInfoGraph<'a> = StableGraph<(Data<'a>, Option<Either<Info, Vec<Info>>>), usize>;

pub fn analyze_subgraph<'a>(
    data_graph: &DataGraph<'a>,
    arg_infos: &[Info],
    nvars: &mut usize,
    reqs: &mut Vec<Condition>,
    uiua: &Uiua,
) -> Result<CompletedInfoGraph<'a>> {
    let roots = data_graph.roots(&uiua.asm);

    let mut info_graph = data_graph.graph.map(|_, &data| (data, None), |_, &x| x);

    for root in roots {
        analyze_node(&mut info_graph, nvars, reqs, root, arg_infos, uiua)?;
    }

    for (_, opt) in info_graph.node_weights() {
        if opt.is_none() {
            bail!("Data graph analysis did not complete");
        }
    }

    // `.unwrap()` should never fail due to the above check
    let graph = info_graph.map_owned(|_, (data, info)| (data, info.unwrap()), |_, x| x);

    Ok(graph)
}

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
    use Either::*;
    let info = match data {
        Data::Arg(i) => Left(arg_infos.get(i).context("Insufficient arg info")?.clone()),
        Data::Out => bail!("`Out` node not handled"),
        Data::Node(Node::Push(val)) => Left(Info {
            typ: typ(val),
            shape: ShapeInfo::Known(val.clone()),
        }),

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

        // -- Iterating Modifiers --
        Data::Node(Node::Mod(Rows, funcs, _span)) => impls::rows(funcs, ctx)?,
        Data::Node(Node::Mod(Table, funcs, _span)) => impls::table(funcs, ctx)?,
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
