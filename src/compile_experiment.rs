use itertools::Itertools;
use melior::{
    dialect::{
        arith, func,
        ods::{tensor, tosa},
        DialectRegistry,
    },
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationLike,
        r#type::{FunctionType, RankedTensorType},
        *,
    },
    pass::{self, PassManager},
    utility::register_all_dialects,
    Context,
};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

use crate::{ArgGraph, CallType};

pub fn test(asm: &uiua::Assembly) {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let location = Location::unknown(&context);

    let mut module = Module::new(location);

    let index_type = Type::index(&context);
    let float_type = Type::float64(&context);
    let tensor_type: Type = RankedTensorType::new(&[3], float_type, None).into();

    let mut arg_graphs = Vec::new();
    for binding in &asm.bindings {
        if let uiua::BindingKind::Func(func) = &binding.kind
            && let uiua::FunctionId::Named(name) = &func.id
        {
            let ag = crate::ArgGraph::from_node(asm, &mut arg_graphs, &asm[func], &[]);
            let op = create_func(
                name,
                &ag,
                &context,
                index_type,
                float_type,
                tensor_type,
                location,
            );
            module.body().append_operation(op);
        }
    }

    assert!(module.as_operation().verify());

    println!("before passes");
    println!("{}", module.as_operation());

    let pass_manager = PassManager::new(&context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow()); // needed because to_llvm doesn't include it.
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.run(&mut module).unwrap();

    println!("after passes");
    println!("{}", module.as_operation());
}

pub fn create_func<'c>(
    name: &str,
    ag: &ArgGraph,
    context: &'c Context,
    index_type: Type<'c>,
    float_type: Type<'c>,
    tensor_type: Type<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    let args_count = ag
        .graph
        .node_weights()
        .filter(|call| matches!(call.inner, CallType::Arg(_)))
        .count();

    // let sig_in = vec![float_type; args_count];
    // let sig_out = vec![float_type; ag.stack.len()];
    let sig_in = vec![tensor_type; args_count];
    let sig_out = vec![tensor_type; ag.stack.len()];

    func::func(
        context,
        StringAttribute::new(context, name),
        TypeAttribute::new(FunctionType::new(context, &sig_in, &sig_out).into()),
        {
            // let args_arr = vec![(float_type, location); args_count];
            let args_arr = vec![(tensor_type, location); args_count];
            let block = Block::new(&args_arr);
            let mut outs = Vec::with_capacity(ag.stack.len());
            let mut map = HashMap::new();
            for &idx in &ag.stack {
                outs.push(node_value(
                    ag,
                    context,
                    index_type,
                    float_type,
                    tensor_type,
                    &block,
                    location,
                    &mut map,
                    idx,
                ));
            }
            block.append_operation(func::r#return(&outs, location));

            let region = Region::new();
            region.append_block(block);
            region
        },
        &[],
        location,
    )
}

fn node_value<'c, 'a>(
    ag: &ArgGraph,
    context: &'c Context,
    index_type: Type<'c>,
    float_type: Type<'c>,
    tensor_type: Type<'c>,
    block: &'a Block<'c>,
    // TODO: uhh actual location?
    location: Location<'c>,
    map: &mut HashMap<NodeIndex, Value<'c, 'a>>,
    idx: NodeIndex,
) -> Value<'c, 'a> {
    if let Some(val) = map.get(&idx) {
        return *val;
    }
    // let mut deps = ag
    //     .graph
    //     .neighbors(idx)
    //     .map(|dep_idx| node_value(ag, block, location, map, dep_idx))
    //     .collect_vec();
    let deps = ag.graph.neighbors(idx);
    let dep_edges = ag.graph.edges(idx);
    let (deps, _dep_edges): (Vec<_>, Vec<usize>) = deps
        .zip(dep_edges.map(|e| e.weight()))
        .sorted_by_key(|(_, e)| *e)
        .unzip();
    let mut deps = deps
        .into_iter()
        .map(|dep_idx| {
            node_value(
                ag,
                context,
                index_type,
                float_type,
                tensor_type,
                block,
                location,
                map,
                dep_idx,
            )
        })
        .collect_vec();

    let call = ag.graph.node_weight(idx).unwrap();
    let out = match call.inner {
        CallType::Arg(i) => block.argument(i).unwrap().into(),
        CallType::Node(uiua::Node::Prim(uiua::Primitive::Add, _span)) => {
            let rhs = deps.pop().unwrap();
            let lhs = deps.pop().unwrap();
            // block
            //     .append_operation(arith::addf(lhs, rhs, location))
            //     // TODO: handle multi output properly
            //     .result(0)
            //     .unwrap()
            //     .into()
            // let op = tensor::GenerateOperationBuilder::new(context, location)
            //     .result(tensor_type)
            //     .dynamic_extents(&[])
            //     .body({
            //         let block = Block::new(&[(index_type, location)]);
            //         let lhs_ext = tensor::ExtractOperationBuilder::new(context, location)
            //             .result(float_type)
            //             .tensor(lhs)
            //             .indices(&[block.argument(0).unwrap().into()])
            //             .build();
            //         let rhs_ext = tensor::ExtractOperationBuilder::new(context, location)
            //             .result(float_type)
            //             .tensor(rhs)
            //             .indices(&[block.argument(0).unwrap().into()])
            //             .build();
            //         let lhs = block.append_operation(lhs_ext.into()).result(0).unwrap();
            //         let rhs = block.append_operation(rhs_ext.into()).result(0).unwrap();
            //         let sum = block
            //             .append_operation(arith::addf(lhs.into(), rhs.into(), location))
            //             .result(0)
            //             .unwrap();
            //         let yield_op = tensor::YieldOperationBuilder::new(context, location)
            //             .value(sum.into())
            //             .build();
            //         block.append_operation(yield_op.into());
            //         // block.append_operation(func::r#return(&[sum.into()], location));
            //         let region = Region::new();
            //         region.append_block(block);
            //         region
            //     })
            //     .build();
            // block.append_operation(op.into()).result(0).unwrap().into()
            let op = tosa::AddOperationBuilder::new(context, location)
                .output(tensor_type)
                .input_1(lhs)
                .input_2(rhs)
                .build();
            block.append_operation(op.into()).result(0).unwrap().into()
        }
        CallType::Node(uiua::Node::Prim(uiua::Primitive::Sub, _span)) => {
            let rhs = deps.pop().unwrap();
            let lhs = deps.pop().unwrap();
            block
                .append_operation(arith::subf(rhs, lhs, location))
                // TODO: handle multi output properly
                .result(0)
                .unwrap()
                .into()
        }
        _ => unimplemented!(),
    };
    map.insert(idx, out);
    out
}

pub fn example() {
    // We need a registry to hold all the dialects
    let registry = DialectRegistry::new();
    // Register all dialects that come with MLIR.
    register_all_dialects(&registry);

    // The MLIR context, like the LLVM one.
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // A location is a debug location like in LLVM, in MLIR all
    // operations need a location, even if its "unknown".
    let location = Location::unknown(&context);

    // A MLIR module is akin to a LLVM module.
    let mut module = Module::new(location);

    // A integer-like type with platform dependent bit width. (like size_t or usize)
    // This is a type defined in the Builtin dialect.
    let index_type = Type::index(&context);

    // Append a `func::func` operation to the body (a block) of the module.
    // This operation accepts a string attribute, which is the name.
    // A type attribute, which contains a function type in this case.
    // Then it accepts a single region, which is where the body
    // of the function will be, this region can have
    // multiple blocks, which is how you may implement
    // control flow within the function.
    // These blocks each can have more operations.
    module.body().append_operation(func::func(
        &context,
        // accepts a StringAttribute which is the function name.
        StringAttribute::new(&context, "add"),
        // A type attribute, defining the function signature.
        TypeAttribute::new(
            FunctionType::new(&context, &[index_type, index_type], &[index_type]).into(),
        ),
        {
            // The first block within the region, blocks accept arguments
            // In regions with control flow, MLIR leverages
            // this structure to implicitly represent
            // the passage of control-flow dependent values without the complex nuances
            // of PHI nodes in traditional SSA representations.
            let block = Block::new(&[(index_type, location), (index_type, location)]);

            // Use the arith dialect to add the 2 arguments.
            let sum = block.append_operation(arith::addi(
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                location,
            ));

            // Return the result using the "func" dialect return operation.
            block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));

            // The Func operation requires a region,
            // we add the block we created to the region and return it,
            // which is passed as an argument to the `func::func` function.
            let region = Region::new();
            region.append_block(block);
            region
        },
        &[],
        location,
    ));

    assert!(module.as_operation().verify());

    println!("{}", module.as_operation());

    let pass_manager = PassManager::new(&context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::conversion::create_tosa_to_linalg());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.run(&mut module).unwrap();

    println!("{}", module.as_operation());

    // let object = unsafe {llvm_compile}
}
