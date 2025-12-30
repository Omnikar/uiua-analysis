use anyhow::{bail, Context, Result};
use petgraph::stable_graph::{NodeIndex, StableGraph};
use smallvec::SmallVec;
use std::collections::HashSet;
use uiua::{Assembly, ImplPrimitive, Node, Primitive};

pub type Stack = SmallVec<[NodeIndex; 16]>;
pub type SmallStack = SmallVec<[NodeIndex; 4]>;

/// A graph structure used to represent the tacit flow of data through a program
#[derive(Default, Debug, Clone)]
pub struct DataGraph<'a> {
    pub graph: StableGraph<Data<'a>, usize>,
    pub stack: Stack,
    pub under_stack: Stack,
    pub arg_count: usize,
}

/// A single unit of a data graph
#[derive(Debug, Clone, Copy)]
pub enum Data<'a> {
    /// A Uiua execution Node
    Node(&'a Node),
    /// An argument to the function represented by the full graph
    Arg(usize),
    /// A single output from a multi-output `Data` instance
    Out,
}

impl<'a> DataGraph<'a> {
    pub fn from_node(node: &'a Node, asm: &Assembly) -> Result<Self> {
        let mut data_graph = Self::default();
        data_graph.process_node(node)?;
        data_graph.prune(asm);
        Ok(data_graph)
    }

    /// Add argument nodes to the graph as necessary to satisfy a minimum stack size
    pub fn extend_args(&mut self, min_args: usize) {
        for _ in 0..min_args.saturating_sub(self.stack.len()) {
            self.stack
                .insert(0, self.graph.add_node(Data::Arg(self.arg_count)));
            self.arg_count += 1;
        }
    }

    /// Checked pop of the top stack value
    fn stack_pop(&mut self) -> Result<NodeIndex> {
        self.stack.pop().context("Inferred too few arguments")
    }

    /// Checked read of the top stack value
    fn stack_top(&self) -> Result<NodeIndex> {
        self.stack
            .last()
            .copied()
            .context("Inferred too few arguments")
    }

    /// Checked read of the nth stack value
    fn stack_n(&self, n: usize) -> Result<NodeIndex> {
        Ok(self.stack[self
            .stack
            .len()
            .checked_sub(n)
            .context("Inferred too few arguments")?])
    }

    /// Recursively build the graph by handling different node types, including processing stack manipulation
    pub fn process_node(&mut self, node: &'a Node) -> Result<()> {
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
            Node::Push(_value) => self.stack.push(self.graph.add_node(Data::Node(node))),
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
                    .map(|sub| (sub.side, sub.n.unwrap_or(1)))
                    .unwrap_or((Left, 0));

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
                for (i, arg) in drain_args(&mut self.stack, sig.args()).rev().enumerate() {
                    // Each edge is given a weight equal to the index of the node it points at in the arguments of the node that depends on it.
                    // So a `Sub` node will have two arrows pointing out of it, the 0 arrow corresponding to the left argument, and the 1 arrow to the right argument.
                    self.graph.add_edge(new, arg, i);
                }

                if sig.outputs() == 1 {
                    self.stack.push(new);
                } else {
                    // For multi-output functions, an `Out` node is added for each output of the function, and the weights of the edges going from the `Out` node to the node for the function are used to indicate which output of the function each node represents.
                    // For instance, an `UnKeep` node will have two `Out` nodes pointing to it. The one with a 0 edge corresponds to the run lengths, and the one with a 1 edge corresponds to the adjacent-deduplication.
                    for i in (0..sig.outputs()).rev() {
                        let out = self.graph.add_node(Data::Out);
                        self.graph.add_edge(out, new, i);
                        self.stack.push(out);
                    }
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
        roots.extend_from_slice(&self.stack);
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
fn drain_args(stack: &mut Stack, num_args: usize) -> impl DoubleEndedIterator<Item = NodeIndex> {
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
