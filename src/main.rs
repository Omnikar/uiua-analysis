use petgraph::{graph::NodeIndex, Graph};
use std::collections::HashSet;
use std::io::Write;

#[derive(Debug)]
enum Node<'a> {
    Node(&'a uiua::Node),
    Arg(usize),
    Out,
}

fn main() {
    let file = std::env::args().nth(1).unwrap();
    let text = std::fs::read_to_string(file).unwrap();
    let asm = uiua::Assembly::from_uasm(&text).unwrap();
    let uiua::BindingKind::Func(ref func) = asm.bindings[0].kind else {
        panic!("oopsie daisies")
    };
    let node = &asm[func];

    let mut arg_graph = ArgGraph::default();

    println!("{:?}", node);

    arg_graph.process_node(node);
    arg_graph.prune(&asm);

    let mut f = std::fs::File::create("graph.dot").unwrap();
    writeln!(
        f,
        "{:?}",
        // petgraph::dot::Dot::with_config(&arg_graph.graph, &[petgraph::dot::Config::EdgeNoLabel]),
        // petgraph::dot::Dot::with_config(
        //     &arg_graph.graph,
        //     &[petgraph::dot::Config::RankDir(petgraph::dot::RankDir::LR)]
        // ),
        petgraph::dot::Dot::new(&arg_graph.graph),
    )
    .unwrap();
}

#[derive(Default)]
struct ArgGraph<'a> {
    graph: Graph<Node<'a>, usize>,
    stack: Vec<NodeIndex>,
    under_stack: Vec<NodeIndex>,
    arg_count: usize,
}

impl<'a> ArgGraph<'a> {
    fn process_node(&mut self, node: &'a uiua::Node) {
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
            uiua::Node::Push(_value) => self.stack.push(self.graph.add_node(Node::Node(node))),
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
            uiua::Node::Mod(uiua::Primitive::Dip, funcs, _span) => {
                assert_eq!(funcs.len(), 1);
                let skipped = self.stack.pop().unwrap();
                self.process_node(&funcs[0].node);
                self.stack.push(skipped);
            }
            uiua::Node::Mod(uiua::Primitive::Gap, funcs, _span) => {
                assert_eq!(funcs.len(), 1);
                self.stack.pop();
                self.process_node(&funcs[0].node);
            }
            uiua::Node::Mod(uiua::Primitive::Fork, funcs, _span) => {
                let reused: Vec<_> = drain_args(&mut self.stack, sig.args()).collect();
                for func in funcs.iter().rev() {
                    self.stack.extend_from_slice(args(&reused, func.sig.args()));
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
                let saved: Vec<_> = drain_args(&mut self.stack, func.sig.args()).collect();
                self.process_node(&func.node);
                self.stack.extend(saved);
                self.process_node(&func.node);
            }
            uiua::Node::ImplMod(uiua::ImplPrimitive::BothImpl(sub), funcs, _span) => {
                use uiua::SubSide::*;

                assert_eq!(funcs.len(), 1);
                let func = &funcs[0];
                let args = func.sig.args();

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
                let to_repeat: Vec<_> = self.stack.drain(repeat_range).collect();

                let mut saved: Vec<_> = drain_args(&mut self.stack, (args - repeat_count) * count)
                    .rev()
                    .collect();
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
                for node in nodes {
                    self.process_node(node);
                }
            }
            node => {
                let new = self.graph.add_node(Node::Node(node));
                for (i, arg) in drain_args(&mut self.stack, sig.args()).rev().enumerate() {
                    self.graph.add_edge(new, arg, i);
                }

                if sig.outputs() == 1 {
                    self.stack.push(new);
                } else {
                    for i in (0..sig.outputs()).rev() {
                        let out = self.graph.add_node(Node::Out);
                        self.graph.add_edge(out, new, i);
                        self.stack.push(out);
                    }
                }
            }
        }
    }

    /// Add argument nodes to the graph as necessary to satisfy a minimum stack size
    fn extend_args(&mut self, min_args: usize) {
        for _ in 0..min_args.saturating_sub(self.stack.len()) {
            self.stack
                .insert(0, self.graph.add_node(Node::Arg(self.arg_count)));
            self.arg_count += 1;
        }
    }

    fn prune(&mut self, asm: &uiua::Assembly) {
        let mut roots: Vec<_> = self
            .graph
            .node_indices()
            .filter(|&idx| {
                if let Some(Node::Node(node)) = self.graph.node_weight(idx) {
                    !node.is_min_purity(uiua::Purity::Impure, asm)
                } else {
                    false
                }
            })
            .collect();
        roots.extend_from_slice(&self.stack);
        let mut unreachable: HashSet<_> = self.graph.node_indices().collect();
        for root in roots {
            let mut bfs = petgraph::visit::Bfs::new(&self.graph, root);
            while let Some(idx) = bfs.next(&self.graph) {
                unreachable.remove(&idx);
            }
        }
        for idx in unreachable {
            self.graph.remove_node(idx);
        }
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
