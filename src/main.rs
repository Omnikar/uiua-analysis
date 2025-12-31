mod analyze;
mod compile_experiment;
mod graph;

use itertools::Itertools;
use petgraph::{graph::NodeIndex, Graph};
use std::collections::HashSet;
use std::io::Write;

fn main() {
    let file = std::env::args().nth(1).unwrap();
    let text = std::fs::read_to_string(file).unwrap();
    let asm = uiua::Assembly::from_uasm(&text).unwrap();
    let mut uiua = uiua::Uiua::with_native_sys();
    uiua.asm = asm;

    // compile::test(&asm);
    // let node = &asm.root;
    // let graph = graph::DataGraph::from_node(&asm, node).unwrap();
    // let mut f = std::fs::File::create("graph.dot").unwrap();
    // let s = format!(
    //     "{:?}",
    //     // petgraph::dot::Dot::with_config(&arg_graph.graph, &[petgraph::dot::Config::EdgeNoLabel]),
    //     // petgraph::dot::Dot::with_config(
    //     //     &arg_graph.graph,
    //     //     &[petgraph::dot::Config::RankDir(petgraph::dot::RankDir::LR)]
    //     // ),
    //     petgraph::dot::Dot::new(&graph.graph),
    // );
    // let s = s.strip_prefix("digraph {\n").unwrap();
    // writeln!(f, "digraph {{").unwrap();
    // writeln!(f, "    node [shape=box]").unwrap();
    // writeln!(f, r#"    node [fontname="Uiua386"]"#).unwrap();
    // writeln!(f, r#"    edge [fontname="Uiua386"]"#).unwrap();
    // write!(f, "{s}").unwrap();

    // let a = Axis::from(3);
    // let mut b = Axis::from(4);
    // *b.term_mut(&[1]) = 1;
    // let mut c = Axis::from(2);
    // *c.term_mut(&[1]) = -1;
    // dbg!(&a, &b, &c, &a + &b, &b + &c, &a * &b, &a * &c, &b * &c);
    // println!("{}", std::mem::size_of::<Axis>());

    use {
        analyze::{axis::Axis, ShapeInfo},
        smallvec::smallvec,
    };
    let data_graph = graph::DataGraph::from_node(&uiua.asm.root, &uiua.asm).unwrap();
    // let shape = [2, 3].map(analyze::axis::Axis::from).into_iter().collect();
    let mut nvars = 0;
    let shape = smallvec![
        Axis::newvar(&mut nvars),
        Axis::newvar(&mut nvars),
        Axis::newvar(&mut nvars),
    ];
    // let shape2 = [4, 5].map(analyze::axis::Axis::from).into_iter().collect();
    let arg_infos = &[
        // analyze::Info {
        //     typ: 0,
        //     shape: ShapeInfo::Known([1, 2, 3].into()),
        //     // shape: ShapeInfo::Ranked(shape),
        //     // shape: ShapeInfo::Unranked {
        //     //     prefix: shape,
        //     //     suffix: shape2,
        //     // },
        // },
        analyze::Info {
            typ: 0,
            shape: ShapeInfo::Ranked(shape),
        },
    ];
    let info_graph = analyze::analyze_graph(&data_graph, arg_infos, &uiua).unwrap();
    dbg!(&info_graph);
    for req in &info_graph.reqs {
        match req {
            analyze::axis::Condition::Or(rels) => {
                for rel in rels {
                    println!(
                        "{} {} 0",
                        rel.expr,
                        match (rel.ineq, rel.inv) {
                            (false, false) => '=',
                            (true, false) => '>',
                            (false, true) => '≠',
                            (true, true) => '≤',
                        }
                    );
                }
            }
        }
    }

    let mut f = std::fs::File::create("graph.dot").unwrap();
    let s = format!("{:?}", petgraph::dot::Dot::new(&info_graph.graph));
    let s = s.strip_prefix("digraph {\n").unwrap();
    writeln!(f, "digraph {{").unwrap();
    writeln!(f, "    node [shape=box]").unwrap();
    writeln!(f, r#"    node [fontname="Uiua386"]"#).unwrap();
    writeln!(f, r#"    edge [fontname="Uiua386"]"#).unwrap();
    write!(f, "{s}").unwrap();
}

fn test() {
    let file = std::env::args().nth(1).unwrap();
    let text = std::fs::read_to_string(file).unwrap();
    let asm = uiua::Assembly::from_uasm(&text).unwrap();

    let mut arg_graphs = Vec::new();

    // let uiua::BindingKind::Func(ref func) = asm.bindings[0].kind else {
    //     panic!("oopsie daisies")
    // };
    // let node = &asm[func];
    let node = &asm.root;
    dbg!(node.span());

    println!("{:?}", node);

    let infos = &[
        Info {
            // value: Some(uiua::Value::from(6)),
            typ: Some(0),
            scalar: Some(true),
            ..Default::default()
        },
        Info {
            // value: Some(uiua::Value::from(6)),
            typ: Some(0),
            scalar: Some(true),
            ..Default::default()
        },
        // Info {
        //     value: Some(uiua::Value::from([1, 2, 3])),
        //     ..Default::default()
        // },
        // Info {
        //     value: Some(uiua::Value::from([4, 5])),
        //     ..Default::default()
        // },
        // Info {
        //     rank: Some(2),
        //     shape_prefix: [2, 3].into(),
        //     ..Default::default()
        // },
        // Info {
        //     rank: Some(2),
        //     shape_prefix: [3, 4].into(),
        //     ..Default::default()
        // },
    ];

    // let mut arg_graph = ArgGraph::default();
    // arg_graph.process_node(node);
    // arg_graph.prune(&asm);
    // arg_graph.fill_infos(
    //     &asm,
    //     infos,
    // );
    let arg_graph = ArgGraph::from_node(&asm, &mut arg_graphs, node, infos);
    // dbg!(&arg_graphs);

    let mut f = std::fs::File::create("graph.dot").unwrap();
    let s = format!(
        "{:?}",
        // petgraph::dot::Dot::with_config(&arg_graph.graph, &[petgraph::dot::Config::EdgeNoLabel]),
        // petgraph::dot::Dot::with_config(
        //     &arg_graph.graph,
        //     &[petgraph::dot::Config::RankDir(petgraph::dot::RankDir::LR)]
        // ),
        petgraph::dot::Dot::new(&arg_graph.graph),
    );
    let s = s.strip_prefix("digraph {\n").unwrap();
    writeln!(f, "digraph {{").unwrap();
    writeln!(f, "    node [shape=box]").unwrap();
    writeln!(f, r#"    node [fontname="Uiua386"]"#).unwrap();
    writeln!(f, r#"    edge [fontname="Uiua386"]"#).unwrap();
    write!(f, "{s}").unwrap();
}

// #[derive(Debug)]
#[derive(Clone)]
pub struct Call<'a> {
    inner: CallType<'a>,
    // TODO: Use a better error type than string
    info: Option<Result<Info, String>>,
}

impl<'a> Call<'a> {
    pub fn new(inner: CallType<'a>) -> Self {
        Self { inner, info: None }
    }

    pub fn node(node: &'a uiua::Node) -> Self {
        Self::new(CallType::Node(node))
    }

    pub fn arg(i: usize) -> Self {
        Self::new(CallType::Arg(i))
    }

    pub fn out() -> Self {
        Self::new(CallType::Out)
    }
}

impl<'a> std::fmt::Debug for Call<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.inner {
            CallType::Node(node) => write!(f, "{node:?}")?,
            CallType::Arg(i) => write!(f, "Arg {i}")?,
            CallType::Out => write!(f, "Out")?,
        }
        if let Some(Ok(ref info)) = self.info {
            if !info.multi.is_empty() {
                write!(f, " | (multiple outputs)")?;
                return Ok(());
            }
            if let Some(ref val) = info.value {
                write!(f, " = {val}")?;
                return Ok(());
            }
            if let Some(typ) = info.typ
            // Probably temporary
                && typ != 0
            {
                let typ_str = match typ {
                    0 => "ℝ",
                    1 => "@",
                    2 => "□",
                    3 => "ℂ",
                    _ => "",
                };
                write!(f, " | {typ_str}")?;
            }
            if let Some(rank) = info.rank {
                if info.shape_prefix.len() < rank {
                    write!(f, " | rank {rank}")?;
                } else if rank == 0 {
                    write!(f, " | scalar")?;
                }
                if !info.shape_prefix.is_empty() {
                    write!(f, " | shape {}", info.shape_prefix)?;
                    if info.shape_prefix.len() < rank {
                        write!(f, "×…")?;
                        if !info.shape_suffix.is_empty() {
                            write!(f, "×{}", info.shape_suffix)?;
                        }
                    }
                } else if !info.shape_suffix.is_empty() {
                    write!(f, " | shape …×{}", info.shape_suffix)?;
                }
            } else {
                if !info.shape_prefix.is_empty() {
                    write!(f, " | shape {}×…", info.shape_prefix)?;
                }
                if !info.shape_suffix.is_empty() {
                    write!(f, " | shape …×{}", info.shape_suffix)?;
                }
            }
        } else if let Some(Err(ref e)) = self.info {
            write!(f, " | {e}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CallType<'a> {
    Node(&'a uiua::Node),
    Arg(usize),
    Out,
}

#[derive(Debug, Clone, Default)]
pub struct Info {
    pub typ: Option<u8>,
    pub rank: Option<usize>,
    pub scalar: Option<bool>,
    pub shape_prefix: uiua::Shape,
    pub shape_suffix: uiua::Shape,
    pub value: Option<uiua::Value>,
    pub multi: Vec<Info>,
}

// impl Info {
//     fn shape(&self) -> Option<&uiua::Shape> {
//     }
// }

#[derive(Default, Debug, Clone)]
pub struct ArgGraph<'a> {
    pub graph: Graph<Call<'a>, usize>,
    pub stack: Vec<NodeIndex>,
    pub under_stack: Vec<NodeIndex>,
    pub arg_count: usize,
}

impl<'a> ArgGraph<'a> {
    pub fn from_node(
        asm: &uiua::Assembly,
        arg_graphs: &mut Vec<(Self, usize)>,
        node: &'a uiua::Node,
        arg_infos: &[Info],
    ) -> Self {
        let mut arg_graph = Self::default();
        arg_graph.process_node(node);
        arg_graph.prune(asm);
        arg_graph.fill_infos(asm, arg_graphs, arg_infos);
        arg_graph
    }

    pub fn process_node(&mut self, node: &'a uiua::Node) {
        let sig = node.sig().unwrap();
        self.extend_args(sig.args());
        match node {
            uiua::Node::CustomInverse(custom_inverse, _span) => {
                self.process_node(&custom_inverse.normal.as_ref().unwrap().node);
            }
            uiua::Node::PushUnder(n, _span) => {
                self.under_stack
                    .extend(drain_args(&mut self.stack, *n).rev());
            }
            uiua::Node::CopyToUnder(n, _span) => {
                self.under_stack.extend(args(&self.stack, *n).iter().rev());
            }
            uiua::Node::PopUnder(n, _span) => {
                self.stack
                    .extend(drain_args(&mut self.under_stack, *n).rev());
            }
            uiua::Node::Push(_value) => self.stack.push(self.graph.add_node(Call::node(node))),
            uiua::Node::Prim(uiua::Primitive::Identity, _span) => {}
            uiua::Node::Prim(uiua::Primitive::Pop, _span) => {
                self.stack.pop();
            }
            uiua::Node::Prim(uiua::Primitive::Dup, _span) => {
                self.stack.push(*self.stack.last().unwrap());
            }
            uiua::Node::Prim(uiua::Primitive::Flip, _span) => {
                args_mut(&mut self.stack, 2).reverse();
            }
            uiua::Node::Mod(uiua::Primitive::On, funcs, _span) => {
                assert_eq!(funcs.len(), 1);
                let preserved = *self.stack.last().unwrap();
                self.process_node(&funcs[0].node);
                self.stack.push(preserved);
            }
            uiua::Node::Mod(uiua::Primitive::Off, funcs, _span) => {
                assert_eq!(funcs.len(), 1);
                let func = &funcs[0];
                let preserved = *self.stack.last().unwrap();
                self.stack
                    .insert(self.stack.len() - func.sig.args(), preserved);
                self.process_node(&func.node);
            }
            uiua::Node::Mod(uiua::Primitive::With, funcs, _span) => {
                assert_eq!(funcs.len(), 1);
                let func = &funcs[0];
                let preserved = self.stack[self.stack.len() - func.sig.args()];
                self.process_node(&func.node);
                self.stack.push(preserved);
            }
            uiua::Node::Mod(uiua::Primitive::Dip, funcs, _span) => {
                assert_eq!(funcs.len(), 1);
                let skipped = self.stack.pop().unwrap();
                self.process_node(&funcs[0].node);
                self.stack.push(skipped);
            }
            uiua::Node::ImplMod(uiua::ImplPrimitive::DipN(n), funcs, _span) => {
                assert_eq!(funcs.len(), 1);
                let preserved = drain_args(&mut self.stack, *n).collect_vec();
                self.process_node(&funcs[0].node);
                self.stack.extend(preserved);
            }
            uiua::Node::Mod(uiua::Primitive::Gap, funcs, _span) => {
                assert_eq!(funcs.len(), 1);
                self.stack.pop();
                self.process_node(&funcs[0].node);
            }
            uiua::Node::Mod(uiua::Primitive::Fork, funcs, _span) => {
                let reused = drain_args(&mut self.stack, sig.args()).collect_vec();
                for func in funcs.iter().rev() {
                    self.stack.extend_from_slice(args(&reused, func.sig.args()));
                    self.process_node(&func.node);
                }
            }
            uiua::Node::Mod(uiua::Primitive::Bracket, funcs, _span) => {
                let mut args = drain_args(&mut self.stack, sig.args()).rev().collect_vec();
                for func in funcs.iter().rev() {
                    self.stack
                        .extend(drain_args(&mut args, func.sig.args()).rev());
                    self.process_node(&func.node);
                }
            }
            uiua::Node::Mod(uiua::Primitive::Below, funcs, _span) => {
                assert_eq!(funcs.len(), 1);
                let start_i = self.stack.len() - sig.args();
                self.stack.extend_from_within(start_i..);
                self.process_node(&funcs[0].node);
            }
            uiua::Node::Mod(uiua::Primitive::Both, funcs, _span) => {
                assert_eq!(funcs.len(), 1);
                let func = &funcs[0];
                let saved = drain_args(&mut self.stack, func.sig.args()).collect_vec();
                self.process_node(&func.node);
                self.stack.extend(saved);
                self.process_node(&func.node);
            }
            uiua::Node::ImplMod(uiua::ImplPrimitive::BothImpl(sub), funcs, _span) => {
                use uiua::SubSide::*;

                assert_eq!(funcs.len(), 1);
                let func = &funcs[0];
                // TODO: should this be max 1?
                let args = func.sig.args().max(1);

                let count = sub.num.unwrap_or(2) as usize;

                let (side, repeat_count) = sub
                    .side
                    .map(|sub| (sub.side, sub.n.unwrap_or(1)))
                    .unwrap_or((Left, 0));

                let len = self.stack.len();
                let repeat_range = if side == Left {
                    len - repeat_count..len
                } else {
                    let end_i = len - (args - repeat_count) * args;
                    end_i - repeat_count..end_i
                };
                let to_repeat = self.stack.drain(repeat_range).collect_vec();

                let mut saved = drain_args(&mut self.stack, (args - repeat_count) * count)
                    .rev()
                    .collect_vec();
                for _ in 0..count {
                    if side == Right {
                        self.stack.extend_from_slice(&to_repeat);
                    }
                    self.stack
                        .extend(drain_args(&mut saved, args - repeat_count).rev());
                    if side == Left {
                        self.stack.extend_from_slice(&to_repeat);
                    }
                    self.process_node(&func.node);
                }
            }
            uiua::Node::Run(nodes) => {
                // dbg!(nodes, node.span());
                for node in nodes {
                    self.process_node(node);
                }
            }
            node => {
                let new = self.graph.add_node(Call::node(node));
                for (i, arg) in drain_args(&mut self.stack, sig.args()).rev().enumerate() {
                    self.graph.add_edge(new, arg, i);
                }

                if sig.outputs() == 1 {
                    self.stack.push(new);
                } else {
                    for i in (0..sig.outputs()).rev() {
                        let out = self.graph.add_node(Call::out());
                        self.graph.add_edge(out, new, i);
                        self.stack.push(out);
                    }
                }
            }
        }
    }

    /// Add argument nodes to the graph as necessary to satisfy a minimum stack size
    pub fn extend_args(&mut self, min_args: usize) {
        for _ in 0..min_args.saturating_sub(self.stack.len()) {
            self.stack
                .insert(0, self.graph.add_node(Call::arg(self.arg_count)));
            self.arg_count += 1;
        }
    }

    /// Current stack values and mutating purity nodes
    pub fn roots(&self, asm: &uiua::Assembly) -> Vec<NodeIndex> {
        let mut roots = self
            .graph
            .node_indices()
            .filter(|&idx| {
                if let Some(CallType::Node(node)) = self.graph.node_weight(idx).map(|n| &n.inner) {
                    !node.is_min_purity(uiua::Purity::Impure, asm)
                } else {
                    false
                }
            })
            .collect_vec();
        roots.extend_from_slice(&self.stack);
        roots
    }

    /// Remove all nodes that are not reachable from either the current stack values or any mutating purity nodes
    pub fn prune(&mut self, asm: &uiua::Assembly) {
        let roots = self.roots(asm);
        let mut unreachable: HashSet<_> = self.graph.node_indices().collect();
        for root in roots {
            let mut bfs = petgraph::visit::Bfs::new(&self.graph, root);
            while let Some(idx) = bfs.next(&self.graph) {
                unreachable.remove(&idx);
            }
        }
        let mut unreachable: Vec<_> = unreachable.into_iter().collect_vec();
        while let Some(idx) = unreachable.pop() {
            // When removing a node, `petgraph` updates the index of the last node to have it take its place
            let last = self.graph.node_indices().next_back();
            self.graph.remove_node(idx);
            for other_idx in self.stack.iter_mut() {
                if Some(*other_idx) == last {
                    *other_idx = idx;
                }
            }
            for other_idx in unreachable.iter_mut() {
                if Some(*other_idx) == last {
                    *other_idx = idx;
                }
            }
        }
    }

    pub fn fill_infos(
        &mut self,
        asm: &uiua::Assembly,
        arg_graphs: &mut Vec<(Self, usize)>,
        arg_infos: &[Info],
    ) {
        let roots = self.roots(asm);
        for root in roots {
            self.fill_info(asm, arg_graphs, root, arg_infos);
        }
    }

    pub fn fill_info(
        &mut self,
        asm: &uiua::Assembly,
        arg_graphs: &mut Vec<(Self, usize)>,
        idx: NodeIndex,
        arg_infos: &[Info],
    ) {
        if self.graph.node_weight(idx).unwrap().info.is_some() {
            return;
        }
        let deps = self.graph.neighbors(idx);
        let dep_edges = self.graph.edges(idx);
        let (deps, dep_edges): (Vec<_>, Vec<usize>) = deps
            .zip(dep_edges.map(|e| e.weight()))
            .sorted_by_key(|(_, e)| *e)
            .unzip();
        let mut dep_infos = Vec::new();
        let mut new_info = Info::default();
        for &dep in &deps {
            self.fill_info(asm, arg_graphs, dep, arg_infos);
        }
        for dep in deps {
            match self
                .graph
                .node_weight(dep)
                .unwrap()
                .info
                .as_ref()
                .unwrap()
                .clone()
            {
                Ok(info) => dep_infos.push(info),
                Err(e) => {
                    self.graph.node_weight_mut(idx).unwrap().info = Some(Err(e));
                    return;
                }
            }
        }
        let call = self.graph.node_weight(idx).unwrap();
        match call.inner {
            CallType::Arg(i) => {
                new_info = arg_infos.get(i).cloned().unwrap_or_default();
            }
            CallType::Out => {
                let i = dep_edges[0];
                new_info = dep_infos[0].multi.get(i).cloned().unwrap_or_default();
            }
            CallType::Node(uiua::Node::Push(val)) => {
                new_info.value = Some(val.clone());
            }
            CallType::Node(uiua::Node::Prim(prim, _span)) if prim.class().is_pervasive() => {
                // TODO: handle shape suffixes for pervasion
                new_info.rank = dep_infos
                    .iter()
                    .map(|info| info.rank)
                    .try_fold(0, |a, b| Some(a.max(b?)));
                let mut new_shape = [].into();
                for info in dep_infos.iter() {
                    new_shape = match pervade_shape(&new_shape, &info.shape_prefix) {
                        Some(sh) => sh,
                        None => {
                            self.graph.node_weight_mut(idx).unwrap().info =
                                Some(Err("Shape mismatch".into()));
                            return;
                        }
                    }
                }
                new_info.shape_prefix = new_shape;
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Fix, _span)) => {
                let dep_info = dep_infos.pop().unwrap();
                new_info.typ = dep_info.typ;
                new_info.rank = dep_info.rank.map(|r| r + 1);
                new_info.shape_prefix = dep_info.shape_prefix;
                new_info.shape_prefix.insert(0, 1);
                new_info.shape_suffix = dep_info.shape_suffix;
                if let Some(ref val) = dep_info.value {
                    let mut val = val.clone();
                    val.fix();
                    new_info.value = Some(val);
                }
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Range, _span)) => {
                let dep_info = dep_infos.pop().unwrap();
                new_info.typ = Some(0);
                if let Some(ref v) = dep_info.value
                // TODO: handle bytes
                    && let Some(v) = v.as_num_array()
                {
                    if let Some(&v) = v.as_scalar() {
                        new_info.rank = Some(1);
                        new_info.shape_prefix = [v as usize].into();
                    } else {
                        let mut shape: uiua::Shape =
                            v.elements().copied().map(|v| v as usize).collect();
                        shape.push(shape.len());
                        new_info.rank = Some(shape.len());
                        new_info.shape_prefix = shape;
                    }
                } else if let Some(true) = dep_info.scalar {
                    new_info.rank = Some(1);
                } else if let Some(1) = dep_info.rank
                    && let Some(len) = dep_info.shape_prefix.first()
                {
                    new_info.rank = Some(*len + 1);
                    new_info.shape_suffix = [*len].into();
                }
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Len, _span)) => {
                let dep_info = dep_infos.pop().unwrap();
                if let Some(&len) = dep_info.shape_prefix.first() {
                    new_info.value = Some(len.into());
                }
                new_info.typ = Some(0);
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Transpose, _span)) => {
                let dep_info = dep_infos.pop().unwrap();
                new_info.typ = dep_info.typ;
                new_info.rank = dep_info.rank;
                if !dep_info.shape_prefix.is_empty() {
                    new_info.shape_prefix = dep_info.shape_prefix[1..].into();
                    new_info.shape_suffix = dep_info.shape_suffix.clone();
                    new_info.shape_suffix.push(dep_info.shape_prefix[0]);
                }
            }
            CallType::Node(uiua::Node::ImplPrim(uiua::ImplPrimitive::TransposeN(n), _span)) => {
                // This is completely wrong I think (it's the old definition of transpose n)
                let n = *n;
                let m = n.unsigned_abs() as usize;
                let dep_info = dep_infos.pop().unwrap();
                new_info.typ = dep_info.typ;
                new_info.rank = dep_info.rank;
                if n > 0 && dep_info.shape_prefix.len() >= m {
                    new_info.shape_prefix = dep_info.shape_prefix[m..].into();
                    new_info.shape_suffix = dep_info.shape_suffix.clone();
                    new_info
                        .shape_suffix
                        .extend_from_slice(&dep_info.shape_prefix[..m]);
                } else if n < 0 && dep_info.shape_suffix.len() >= m {
                    let split = dep_info.shape_suffix.len() - m;
                    new_info.shape_suffix = dep_info.shape_suffix[..split].into();
                    new_info.shape_prefix = dep_info.shape_suffix[split..].into();
                    new_info
                        .shape_prefix
                        .extend_from_slice(&dep_info.shape_prefix);
                }
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Deshape, _span)) => {
                new_info.rank = Some(1);
                // TODO: New shape if entire shape is known
            }
            CallType::Node(uiua::Node::ImplPrim(uiua::ImplPrimitive::DeshapeSub(sub), _span)) => {
                if *sub >= 0 {
                    new_info.rank = Some(*sub as usize);
                } else {
                    new_info.rank = dep_infos[0]
                        .rank
                        .map(|r| r.saturating_add_signed(*sub as isize));
                }
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Gen, _span)) => {
                let left_info = dep_infos.remove(0);
                new_info.typ = Some(0);
                if let Some(len) = left_info.shape_prefix.first() {
                    new_info.rank = Some(*len);
                }
                if let Some(ref val) = left_info.value {
                    let sh: uiua::Shape = if let Some(val) = val.as_num_array() {
                        val.elements().map(|v| *v as usize).collect()
                    } else if let Some(val) = val.as_byte_array() {
                        val.elements().map(|v| *v as usize).collect()
                    } else {
                        self.graph.node_weight_mut(idx).unwrap().info =
                            Some(Err("Gen shape requires numbers".into()));
                        return;
                    };
                    if val.shape.len() > 1 {
                        self.graph.node_weight_mut(idx).unwrap().info =
                            Some(Err("Gen passed a rank >1 shape".into()));
                        return;
                    }
                    new_info.rank = Some(sh.len());
                    new_info.shape_prefix = sh;
                }
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Reshape, _span)) => {
                let right_info = dep_infos.pop().unwrap();
                let left_info = dep_infos.pop().unwrap();
                new_info.typ = right_info.typ;
                // TODO: Do better error checking here
                if let Some(true) = left_info.scalar
                    && let Some(val) = left_info.value
                {
                    // let Some(val) = val.as_num_array().and_then(|v| v.as_scalar()) else {
                    //     self.graph.node_weight_mut(idx).unwrap().info =
                    //         Some(Err("Reshape requires numbers".into()));
                    //     return;
                    // };
                    let val = if let Some(val) = val.as_num_array().and_then(|v| v.as_scalar()) {
                        *val as usize
                    } else if let Some(val) = val.as_byte_array().and_then(|v| v.as_scalar()) {
                        *val as usize
                    } else {
                        self.graph.node_weight_mut(idx).unwrap().info =
                            Some(Err("Reshape requires numbers".into()));
                        return;
                    };
                    let mut sh = right_info.shape_prefix;
                    sh.insert(0, val);
                    new_info.shape_prefix = sh;
                    new_info.rank = right_info.rank.map(|r| r + 1);
                } else if let Some(false) = left_info.scalar {
                    if let Some(val) = left_info.value {
                        let sh: uiua::Shape = if let Some(val) = val.as_num_array() {
                            val.elements().map(|v| *v as usize).collect()
                        } else if let Some(val) = val.as_byte_array() {
                            val.elements().map(|v| *v as usize).collect()
                        } else {
                            self.graph.node_weight_mut(idx).unwrap().info =
                                Some(Err("Reshape requires numbers".into()));
                            return;
                        };
                        if val.shape.len() > 1 {
                            self.graph.node_weight_mut(idx).unwrap().info =
                                Some(Err("Reshape passed a rank >1 shape".into()));
                            return;
                        }
                        new_info.rank = Some(sh.len());
                        new_info.shape_prefix = sh;
                    } else if let Some(len) = left_info.shape_prefix.first() {
                        // TODO: error if passing a rank >1 left argument
                        new_info.rank = Some(*len);
                    }
                }
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Select, _span)) => {
                let right_info = dep_infos.pop().unwrap();
                let left_info = dep_infos.pop().unwrap();
                if let Some(lr) = left_info.rank
                    && let Some(rr) = right_info.rank
                {
                    new_info.rank = Some(lr + rr.max(1) - 1)
                }
                new_info.shape_prefix = left_info.shape_prefix.clone();
                if Some(left_info.shape_prefix.len()) == left_info.rank {
                    new_info.shape_prefix.extend_from_slice(
                        &right_info.shape_prefix[right_info.shape_prefix.len().min(1)..],
                    );
                }
                // TODO: shape suffix handling
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Keep, _span)) => {
                // TODO: length based on sum of left arg? make sure to account for repeating behavior when left arg is too short
                new_info.rank = dep_infos[1].rank;
            }
            CallType::Node(uiua::Node::ImplPrim(uiua::ImplPrimitive::MultiKeep(_), _span)) => {
                new_info.rank = dep_infos[1].rank;
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Take, _span)) => {
                new_info.rank = dep_infos[1].rank;
                // TODO: shape
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Drop, _span)) => {
                new_info.rank = dep_infos[1].rank;
                // TODO: shape
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Couple, _span)) => {
                // new_info.rank = dep_infos
                //     .iter()
                //     .map(|info| info.rank)
                //     .try_fold(0, |a, b| Some(a.max(b?)))
                //     .map(|r| r + 1);
                let right_info = dep_infos.pop().unwrap();
                let left_info = dep_infos.pop().unwrap();
                if let Some(true) = left_info.scalar
                    && let Some(true) = right_info.scalar
                    && let Some(ref left) = left_info.value
                    && let Some(ref right) = right_info.value
                {
                    if let Some(left) = left.as_num_array()
                        && let Some(left) = left.as_scalar()
                        && let Some(right) = right.as_num_array()
                        && let Some(right) = right.as_scalar()
                    {
                        new_info.value = Some([*left, *right].into());
                    } else if let Some(left) = left.as_num_array()
                        && let Some(left) = left.as_scalar()
                        && let Some(right) = right.as_byte_array()
                        && let Some(right) = right.as_scalar()
                    {
                        new_info.value = Some([*left, *right as f64].into());
                    } else if let Some(left) = left.as_byte_array()
                        && let Some(left) = left.as_scalar()
                        && let Some(right) = right.as_num_array()
                        && let Some(right) = right.as_scalar()
                    {
                        new_info.value = Some([*left as f64, *right].into());
                    } else if let Some(left) = left.as_byte_array()
                        && let Some(left) = left.as_scalar()
                        && let Some(right) = right.as_byte_array()
                        && let Some(right) = right.as_scalar()
                    {
                        new_info.value = Some([*left, *right].into());
                    }
                } else {
                    new_info.rank = left_info
                        .rank
                        .and_then(|lr| right_info.rank.map(|rr| lr.max(rr) + 1));
                    // TODO more shape stuff
                    new_info.shape_prefix.insert(0, 2);
                }
            }
            CallType::Node(uiua::Node::ImplPrim(uiua::ImplPrimitive::UnCouple, _span)) => {
                let dep_info = dep_infos.pop().unwrap();
                if let Some(true) = dep_info.scalar {
                    self.graph.node_weight_mut(idx).unwrap().info =
                        Some(Err("Cannot uncouple scalar".into()));
                    return;
                }
                if let Some(len) = dep_info.shape_prefix.first()
                    && *len != 2
                {
                    self.graph.node_weight_mut(idx).unwrap().info =
                        Some(Err(format!("Cannot uncouple array with {len} rows")));
                    return;
                }
                let out_info = Info {
                    typ: dep_info.typ,
                    rank: dep_info.rank.map(|r| r.saturating_sub(1)),
                    shape_prefix: dep_info.shape_prefix[1..].into(),
                    // TODO: shape suffix
                    ..Default::default()
                };
                new_info.multi.push(out_info.clone());
                new_info.multi.push(out_info);
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::MemberOf, _span)) => {
                let right_info = dep_infos.pop().unwrap();
                let left_info = dep_infos.pop().unwrap();
                // Formula found experimentally ¯\_(ツ)_/¯
                new_info.rank = left_info
                    .rank
                    .and_then(|lr| right_info.rank.map(|rr| (rr + 1).abs_diff(lr.max(1))));
                // TODO: shape
            }
            CallType::Node(uiua::Node::ImplPrim(uiua::ImplPrimitive::FirstMinIndex, _span)) => {
                new_info.typ = Some(0);
                new_info.scalar = Some(true);
            }
            CallType::Node(uiua::Node::ImplPrim(uiua::ImplPrimitive::Primes, _span)) => {
                new_info.rank = dep_infos[0].rank.map(|r| r + 1);
                new_info.shape_suffix = dep_infos[0].shape_suffix.clone();
            }
            CallType::Node(uiua::Node::ImplPrim(uiua::ImplPrimitive::UnKeep, _span)) => {
                let dep_info = &dep_infos[0];
                new_info.multi = vec![dep_info.clone(), dep_info.clone()];
            }
            CallType::Node(uiua::Node::Prim(uiua::Primitive::Rand, _span)) => {
                new_info.typ = Some(0);
                new_info.scalar = Some(true);
            }
            CallType::Node(uiua::Node::Mod(uiua::Primitive::Rows, funcs, _span)) => {
                let mut new_len = Some(1);
                for (sh, rank) in dep_infos.iter().map(|info| (&info.shape_prefix, info.rank)) {
                    if let Some(&len) = sh.first()
                        && let Some(ref mut new_len) = new_len
                    {
                        if *new_len == 1 {
                            *new_len = len;
                        } else if len != 1 && *new_len != len {
                            self.graph.node_weight_mut(idx).unwrap().info =
                                Some(Err("Row count mismatch".into()));
                            return;
                        }
                    } else if rank != Some(0) {
                        new_len = None;
                    }
                }
                let mut new_dep_infos = Vec::new();
                for info in dep_infos.iter() {
                    let value = if info.shape_prefix.first().copied().unwrap_or(1) == 1 {
                        info.value.as_ref().map(|val| {
                            let mut val = val.clone();
                            if !val.shape.is_empty() {
                                val.shape = val.shape[1..].into();
                            }
                            val
                        })
                    } else {
                        None
                    };
                    new_dep_infos.push(Info {
                        typ: info.typ,
                        rank: info.rank.map(|r| r.saturating_sub(1)),
                        scalar: info.rank.map(|r| r <= 1),
                        shape_prefix: info.shape_prefix[info.shape_prefix.len().min(1)..].into(),
                        // shape_suffix: TODO
                        value,
                        ..Default::default()
                    });
                }

                // let mut arg_graph = ArgGraph::default();
                // arg_graph.process_node(&funcs[0].node);
                // arg_graph.fill_infos(asm, &new_dep_infos);
                let node = &funcs[0].node;
                let arg_graph = ArgGraph::from_node(asm, arg_graphs, node, &new_dep_infos);
                if let Some(span) = node.span() {
                    arg_graphs.push((arg_graph.clone(), span));
                }
                let process_info = |mut info: Info| {
                    info.rank = info.rank.map(|r| r + 1);
                    if let Some(new_len) = new_len {
                        info.shape_prefix.insert(0, new_len);
                    }
                    info
                };
                if arg_graph.stack.len() == 1 {
                    let info_res = arg_graph
                        .graph
                        .node_weight(arg_graph.stack[0])
                        .unwrap()
                        .info
                        .as_ref()
                        .unwrap();
                    match info_res {
                        Ok(info) => new_info = process_info(info.clone()),
                        Err(e) => {
                            self.graph.node_weight_mut(idx).unwrap().info = Some(Err(e.clone()));
                            return;
                        }
                    }
                } else if arg_graph.stack.len() > 1 {
                    for idx in arg_graph.stack.into_iter().rev() {
                        let info_res = arg_graph
                            .graph
                            .node_weight(idx)
                            .unwrap()
                            .info
                            .as_ref()
                            .unwrap();
                        match info_res {
                            Ok(info) => new_info.multi.push(process_info(info.clone())),
                            Err(e) => {
                                self.graph.node_weight_mut(idx).unwrap().info =
                                    Some(Err(e.clone()));
                                return;
                            }
                        }
                    }
                }
            }
            CallType::Node(uiua::Node::Mod(uiua::Primitive::Table, funcs, _span)) => {
                let mut new_shape_prefix = uiua::Shape::default();
                let mut shape_prefix_incomplete = false;
                let mut new_dep_infos = Vec::new();
                for info in dep_infos.iter() {
                    if !shape_prefix_incomplete {
                        if let Some(len) = info.shape_prefix.first() {
                            new_shape_prefix.push(*len);
                        } else if info.rank == Some(0) {
                            new_shape_prefix.push(1);
                        } else {
                            shape_prefix_incomplete = true;
                        }
                    }

                    let value = if info.shape_prefix.first().copied().unwrap_or(1) == 1 {
                        info.value.as_ref().map(|val| {
                            let mut val = val.clone();
                            if !val.shape.is_empty() {
                                val.shape = val.shape[1..].into();
                            }
                            val
                        })
                    } else {
                        None
                    };
                    new_dep_infos.push(Info {
                        typ: info.typ,
                        rank: info.rank.map(|r| r.saturating_sub(1)),
                        scalar: info.rank.map(|r| r <= 1),
                        shape_prefix: info.shape_prefix[info.shape_prefix.len().min(1)..].into(),
                        // shape_suffix: TODO
                        value,
                        ..Default::default()
                    });
                }

                // let mut arg_graph = ArgGraph::default();
                // arg_graph.process_node(&funcs[0].node);
                // arg_graph.fill_infos(asm, arg_graphs, &new_dep_infos);
                let node = &funcs[0].node;
                let arg_graph = ArgGraph::from_node(asm, arg_graphs, node, &new_dep_infos);
                if let Some(span) = node.span() {
                    arg_graphs.push((arg_graph.clone(), span));
                }
                let process_info = |mut info: Info| {
                    // if shape_prefix_incomplete {
                    //     info.rank = None;
                    // } else {
                    //     info.rank = info.rank.map(|r| r + new_shape_prefix.len());
                    // }
                    info.rank = info.rank.map(|r| r + dep_infos.len());

                    let after = info.shape_prefix;
                    info.shape_prefix = new_shape_prefix.clone();
                    if !shape_prefix_incomplete {
                        info.shape_prefix.extend(after);
                    }
                    info
                };
                if arg_graph.stack.len() == 1 {
                    let info_res = arg_graph
                        .graph
                        .node_weight(arg_graph.stack[0])
                        .unwrap()
                        .info
                        .as_ref()
                        .unwrap();
                    match info_res {
                        Ok(info) => new_info = process_info(info.clone()),
                        Err(e) => {
                            self.graph.node_weight_mut(idx).unwrap().info = Some(Err(e.clone()));
                            return;
                        }
                    }
                } else if arg_graph.stack.len() > 1 {
                    for idx in arg_graph.stack.into_iter().rev() {
                        let info_res = arg_graph
                            .graph
                            .node_weight(idx)
                            .unwrap()
                            .info
                            .as_ref()
                            .unwrap();
                        match info_res {
                            Ok(info) => new_info.multi.push(process_info(info.clone())),
                            Err(e) => {
                                self.graph.node_weight_mut(idx).unwrap().info =
                                    Some(Err(e.clone()));
                                return;
                            }
                        }
                    }
                }
            }
            CallType::Node(uiua::Node::Mod(uiua::Primitive::Reduce, funcs, _span)) => {
                let dep_info = dep_infos.pop().unwrap();
                new_info.typ = dep_info.typ;
                new_info.rank = dep_info.rank.map(|r| r.saturating_sub(1));
                new_info.shape_prefix =
                    dep_info.shape_prefix[dep_info.shape_prefix.len().min(1)..].into();
                // TODO: propagate arg info
                let node = &funcs[0].node;
                let arg_graph = ArgGraph::from_node(asm, arg_graphs, node, &[]);
                if let Some(span) = node.span() {
                    arg_graphs.push((arg_graph, span));
                }
            }
            CallType::Node(uiua::Node::ImplMod(
                uiua::ImplPrimitive::ReduceDepth(depth),
                funcs,
                _span,
            )) => {
                let dep_info = dep_infos.pop().unwrap();
                new_info.typ = dep_info.typ;
                new_info.rank = dep_info.rank.map(|r| r.saturating_sub(1));
                new_info.shape_prefix = dep_info.shape_prefix;
                if new_info.shape_prefix.len() > *depth {
                    new_info.shape_prefix.remove(*depth);
                }

                // TODO: propagate arg info
                let node = &funcs[0].node;
                let arg_graph = ArgGraph::from_node(asm, arg_graphs, node, &[]);
                if let Some(span) = node.span() {
                    arg_graphs.push((arg_graph, span));
                }
            }
            _ => {}
        }
        if let Some(ref val) = new_info.value {
            let shape = val.shape.clone();
            let rank = shape.len();
            new_info.rank = Some(rank);
            new_info.shape_prefix = shape;
            new_info.typ = Some(match val {
                uiua::Value::Byte(_) | uiua::Value::Num(_) => 0,
                uiua::Value::Char(_) => 1,
                uiua::Value::Box(_) => 2,
                uiua::Value::Complex(_) => 3,
            });
        }
        if let Some(rank) = new_info.rank {
            new_info.scalar = Some(rank == 0);

            // If the shape prefix and suffix are large enough to be known to cover the full shape, expand both to be the full shape
            if new_info.shape_prefix.len() + new_info.shape_suffix.len() >= rank {
                new_info.shape_prefix.truncate(rank);
                if new_info.shape_suffix.len() > rank {
                    new_info
                        .shape_suffix
                        .drain(0..new_info.shape_suffix.len() - rank);
                }
                // TODO: check that the overlap matches and emit an error otherwise
                new_info.shape_prefix.extend_from_slice(
                    &new_info.shape_suffix
                        [new_info.shape_suffix.len() + new_info.shape_prefix.len() - rank..],
                );
                new_info.shape_suffix = new_info.shape_prefix.clone();
            }
        } else if let Some(true) = new_info.scalar {
            new_info.rank = Some(0);
        }
        self.graph.node_weight_mut(idx).unwrap().info = Some(Ok(new_info));
    }
}

/// Draining iterator over the top `num_args` items in push-order
fn drain_args(
    stack: &mut Vec<NodeIndex>,
    num_args: usize,
) -> impl DoubleEndedIterator<Item = NodeIndex> {
    stack.drain(stack.len() - num_args..)
}

/// Slice of the top `num_args` items in push-order
fn args(stack: &[NodeIndex], num_args: usize) -> &[NodeIndex] {
    &stack[stack.len() - num_args..]
}

/// Slice of the top `num_args` items in push-order
fn args_mut(stack: &mut [NodeIndex], num_args: usize) -> &mut [NodeIndex] {
    let len = stack.len();
    &mut stack[len - num_args..]
}

fn pervade_shape(a: &uiua::Shape, b: &uiua::Shape) -> Option<uiua::Shape> {
    a.iter()
        .copied()
        .zip_longest(b.iter().copied())
        .map(|p| p.or(1, 1))
        .map(|(a, b)| {
            if a == 1 || a == b {
                Some(b)
            } else if b == 1 {
                Some(a)
            } else {
                None
            }
        })
        .collect()
}
