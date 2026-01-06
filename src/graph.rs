use anyhow::{bail, Context, Result};
use petgraph::stable_graph::{NodeIndex, StableGraph};
use smallvec::SmallVec;
use std::collections::HashSet;
use uiua::{Assembly, ImplPrimitive, Node, Primitive};

pub type Stack = SmallVec<[(NodeIndex, usize); 4]>;

/// A graph structure used to represent the tacit flow of data through a program
#[derive(Default, Debug, Clone)]
pub struct DataGraph<'u> {
    pub graph: StableGraph<Data<'u>, (usize, usize)>,
    pub stack: Stack,
    pub under_stack: Stack,
    pub arg_count: usize,
}

/// A single unit of a data graph
#[derive(Debug, Clone, Copy)]
pub enum Data<'u> {
    /// A Uiua execution Node
    Node(&'u Node),
    /// An argument to the function represented by the full graph
    Arg(usize),
}

impl<'u> DataGraph<'u> {
    pub fn from_node(node: &'u Node, asm: &Assembly) -> Result<Self> {
        let mut data_graph = Self::default();
        data_graph.process_node(node)?;
        data_graph.prune(asm);
        Ok(data_graph)
    }

    /// Add argument nodes to the graph as necessary to satisfy a minimum stack size
    pub fn extend_args(&mut self, min_args: usize) {
        for _ in 0..min_args.saturating_sub(self.stack.len()) {
            self.stack
                .insert(0, (self.graph.add_node(Data::Arg(self.arg_count)), 0));
            self.arg_count += 1;
        }
    }

    /// Checked pop of the top stack value
    fn stack_pop(&mut self) -> Result<(NodeIndex, usize)> {
        self.stack.pop().context("Inferred too few arguments")
    }

    /// Checked read of the top stack value
    fn stack_top(&self) -> Result<(NodeIndex, usize)> {
        self.stack
            .last()
            .copied()
            .context("Inferred too few arguments")
    }

    /// Checked read of the nth stack value
    fn stack_n(&self, n: usize) -> Result<(NodeIndex, usize)> {
        Ok(self.stack[self
            .stack
            .len()
            .checked_sub(n)
            .context("Inferred too few arguments")?])
    }

    /// Recursively build the graph by handling different node types, including processing stack manipulation
    pub fn process_node(&mut self, node: &'u Node) -> Result<()> {
        let sig = node.sig().ok().context("Failed to get node signature")?;
        self.extend_args(sig.args());

        /// Used to error if a modifier was passed any amount of functions other than one
        fn one_func(prim: Primitive, funcs: &[uiua::SigNode]) -> Result<&uiua::SigNode> {
            if funcs.len() != 1 {
                bail!(
                    "{} passed {} functions instead of 1",
                    prim.format(),
                    funcs.len()
                );
            }
            Ok(&funcs[0])
        }

        // This is one big `match` block that comprises all the supported stack manipulation operations, ending with a catchall branch that is applied to anything that doesn't fall into the stack manipulation operations and other node types supported here
        use ImplPrimitive::*;
        use Primitive::*;
        match node {
            Node::CustomInverse(custom_inverse, _span) => {
                let node = custom_inverse
                    .normal
                    .as_ref()
                    .ok()
                    .context("No inverse found")?;
                self.process_node(&node.node)?;
            }
            Node::PushUnder(n, _span) => self
                .under_stack
                .extend(drain_args(&mut self.stack, *n).rev()),
            Node::CopyToUnder(n, _span) => self
                .under_stack
                .extend(args(&self.stack, *n).iter().copied().rev()),
            Node::PopUnder(n, _span) => self
                .stack
                .extend(drain_args(&mut self.under_stack, *n).rev()),
            Node::Prim(Identity, _span) => {}
            Node::Prim(Pop, _span) => {
                self.stack.pop();
            }
            Node::Prim(Flip, _span) => args_mut(&mut self.stack, 2).reverse(),
            Node::Mod(On, funcs, _span) => {
                let func = one_func(On, funcs)?;
                let preserved = self.stack_top()?;
                self.process_node(&func.node)?;
                self.stack.push(preserved);
            }
            Node::Mod(Off, funcs, _span) => {
                let func = one_func(Off, funcs)?;
                let preserved = self.stack_top()?;
                self.stack
                    .insert(self.stack.len() - func.sig.args(), preserved);
                self.process_node(&func.node)?;
            }
            Node::Mod(With, funcs, _span) => {
                let func = one_func(With, funcs)?;
                let preserved = self.stack_n(func.sig.args())?;
                self.process_node(&func.node)?;
                self.stack.push(preserved);
            }
            Node::Mod(Dip, funcs, _span) => {
                let func = one_func(Dip, funcs)?;
                let skipped = self.stack_pop()?;
                self.process_node(&func.node)?;
                self.stack.push(skipped);
            }
            Node::ImplMod(DipN(n), funcs, _span) => {
                let func = one_func(Dip, funcs)?;
                let preserved: Stack = drain_args(&mut self.stack, *n).collect();
                self.process_node(&func.node)?;
                self.stack.extend(preserved);
            }
            Node::Mod(Fork, funcs, _span) => {
                let reused: Stack = drain_args(&mut self.stack, sig.args()).collect();
                for func in funcs.iter().rev() {
                    self.stack.extend_from_slice(args(&reused, func.sig.args()));
                    self.process_node(&func.node)?;
                }
            }
            Node::Mod(Bracket, funcs, _span) => {
                let mut args: Stack = drain_args(&mut self.stack, sig.args()).rev().collect();
                for func in funcs.iter().rev() {
                    self.stack
                        .extend(drain_args(&mut args, func.sig.args()).rev());
                    self.process_node(&func.node)?;
                }
            }
            Node::Mod(Below, funcs, _span) => {
                let func = one_func(Below, funcs)?;
                let start_i = self.stack.len() - sig.args();
                self.stack.extend_from_slice(&self.stack.clone()[start_i..]);
                self.process_node(&func.node)?;
            }
            Node::Mod(Both, funcs, _span) => {
                let func = one_func(Both, funcs)?;
                let saved: Stack = drain_args(&mut self.stack, func.sig.args()).collect();
                self.process_node(&func.node)?;
                self.stack.extend(saved);
                self.process_node(&func.node)?;
            }
            Node::ImplMod(ImplPrimitive::BothImpl(sub), funcs, _span) => {
                use uiua::SubSide::*;

                let func = one_func(Both, funcs)?;
                // TODO: should this be max 1?
                let args = func.sig.args().max(1);

                let count = sub.num.unwrap_or(2) as usize;

                let (side, repeat_count) = sub
                    .side
                    .map_or((Left, 0), |sub| (sub.side, sub.n.unwrap_or(1)));

                let len = self.stack.len();
                let repeat_range = if side == Left {
                    len - repeat_count..len
                } else {
                    let end_i = len - (args - repeat_count) * args;
                    end_i - repeat_count..end_i
                };
                let to_repeat: Stack = self.stack.drain(repeat_range).collect();

                let mut saved: Stack = drain_args(&mut self.stack, (args - repeat_count) * count)
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
                    self.process_node(&func.node)?;
                }
            }
            Node::Run(nodes) => {
                for node in nodes {
                    self.process_node(node)?;
                }
            }
            // This is the branch that actually creates the main nodes, connecting each one to the appropriate number of inputs from the stack
            node => {
                let new = self.graph.add_node(Data::Node(node));
                for (in_i, (arg, out_i)) in
                    drain_args(&mut self.stack, sig.args()).rev().enumerate()
                {
                    // Each edge is given a weight consisting of a tuple of two numbers. The first number indicates which output from the depended-upon node is being used, and the second number indicates which input for the dependent node it is used for.
                    // So a `Sub` node will have two arrows pointing out of it, one arrow will have weight (_, 0), corresponding to the left argument, and the other arrow will have weight (_, 1), corresponding to the right argument.
                    // As another example, consider an `UnKeep` node. An arrow pointing toward it with weight (0, _) indicates that something is using the run-length output, whereas an arrow pointing toward it with weight (1, _) indicates that something is using the adjacent-deduplicated output.
                    self.graph.add_edge(new, arg, (out_i, in_i));
                }

                for out_i in (0..sig.outputs()).rev() {
                    self.stack.push((new, out_i));
                }
            }
        }

        Ok(())
    }

    /// Current stack values and mutating purity nodes
    /// Anything not reachable from a root is considered dead code
    pub fn roots(&self, asm: &Assembly) -> Vec<NodeIndex> {
        let mut roots: Vec<_> = self
            .graph
            .node_indices()
            .filter(|&idx| {
                if let Some(Data::Node(node)) = self.graph.node_weight(idx) {
                    !node.is_min_purity(uiua::Purity::Impure, asm)
                } else {
                    false
                }
            })
            .collect();
        // NOTE: This might contain duplicates. Is this a problem?
        roots.extend(self.stack.iter().map(|(idx, _)| *idx));
        roots
    }

    /// Remove all nodes that are not reachable from either the current stack values or any mutating purity nodes
    pub fn prune(&mut self, asm: &Assembly) {
        let roots = self.roots(asm);
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
    stack: &mut Stack,
    num_args: usize,
) -> impl DoubleEndedIterator<Item = (NodeIndex, usize)> {
    stack.drain(stack.len() - num_args..)
}

/// Slice of the top `num_args` items in push-order
fn args(stack: &[(NodeIndex, usize)], num_args: usize) -> &[(NodeIndex, usize)] {
    &stack[stack.len() - num_args..]
}

/// Slice of the top `num_args` items in push-order
fn args_mut(stack: &mut [(NodeIndex, usize)], num_args: usize) -> &mut [(NodeIndex, usize)] {
    let len = stack.len();
    &mut stack[len - num_args..]
}
