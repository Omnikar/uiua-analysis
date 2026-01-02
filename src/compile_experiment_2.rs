use anyhow::{bail, Context as AnyhowContext, Result};
use itertools::{Either, Itertools};
use melior::{
    dialect::{
        arith, func,
        ods::{tensor, tosa},
        DialectRegistry,
    },
    ir::{
        attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute},
        operation::OperationLike,
        r#type::{FunctionType, RankedTensorType},
        *,
    },
    pass::{self, PassManager},
    utility::register_all_dialects,
    Context,
};
use petgraph::{data::DataMap, graph::NodeIndex, stable_graph::StableGraph};
use std::collections::HashMap;
use uiua::Node;

use crate::{
    analyze::{analyze_func_graph, axis::Axis, FuncLib, Info, Infos, ShapeInfo},
    graph::{Data, DataGraph},
};

/// The integer used to indicate a dynamic axis length to MLIR
const DYN_AX: u64 = i64::MAX as u64 + 1;

#[derive(Clone, Copy)]
struct CompileContext<'c, 'u> {
    context: &'c Context,
    float_type: Type<'c>,
    uiua: &'u uiua::Uiua,
}

struct CompiledFuncLib<'c> {
    funcs: Vec<Operation<'c>>,
}

// type WorkingCompileGraph<'c, 'a> =
//     StableGraph<(Data<'a>, Option<Either<Value<'c, 'a>, Vec<Value<'c, 'a>>>>), usize>;
type FuncCompileGraph<'c, 'a> = StableGraph<(Data<'a>, Option<Value<'c, 'a>>), usize>;

pub fn compile_test(uiua: &uiua::Uiua) -> Result<()> {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let float_type = Type::float64(&context);
    let ctx = CompileContext {
        context: &context,
        float_type,
        uiua,
    };

    let mut module = Module::new(Location::unknown(&context));

    let mut funclib = FuncLib::new();

    let data_graph = DataGraph::from_node(&uiua.asm.root, &uiua.asm)?;

    let infos = analyze_func_graph(&data_graph, &[], &mut funclib, uiua)?;

    let func_id = uiua::FunctionId::Named("main".into());
    // span 0 smh
    funclib.funcs.push(crate::analyze::AnalyzedFunc::new(
        func_id, data_graph, infos, 0,
    ));

    for i in 0..funclib.funcs.len() {
        let func = compile_func(&funclib, i, ctx)?;
        module.body().append_operation(func);
    }

    println!("{}", module.as_operation());
    assert!(module.as_operation().verify());

    println!("{}", module.as_operation());

    Ok(())
}

fn compile_func<'c>(
    funclib: &FuncLib,
    idx: usize,
    ctx: CompileContext<'c, '_>,
) -> Result<Operation<'c>> {
    let func = &funclib.funcs[idx];
    // TODO: Change the function name based on the inputs
    let func_name = match &func.id {
        uiua::FunctionId::Named(ident) => ident,
        _ => todo!("Non named function in funclib"),
    };

    let loc = span_to_loc(func.span, ctx);

    let mut compile_graph: FuncCompileGraph =
        func.graph.graph.map(|_, &data| (data, None), |_, &x| x);

    let mut sig_in = Vec::new();
    let mut arg_types = Vec::new();
    for arg_info in &func.infos.args {
        if arg_info.typ != 0 {
            bail!("Only numbers are supported currently");
        }
        let arg_type = mk_tensor_type(&arg_info.shape, ctx);
        sig_in.push(arg_type);
        arg_types.push((arg_type, loc));
    }

    let mut sig_out = Vec::new();
    for out_info in &func.infos.outs {
        if out_info.typ != 0 {
            bail!("Only numbers are supported currently");
        }
        let out_type = mk_tensor_type(&out_info.shape, ctx);
        sig_out.push(out_type);
    }

    let block = Block::new(&arg_types);
    // let mut outs = Vec::with_capacity(func.infos.outs.len());

    for root in func.graph.roots(&ctx.uiua.asm) {
        compile_node(root, &block, &mut compile_graph, &func.infos.map, ctx)?;
    }

    let outs = func
        .graph
        .stack
        .iter()
        .map(|&idx| compile_graph.node_weight(idx).and_then(|(_, v)| *v))
        .collect::<Option<Vec<_>>>()
        .context("Did not compile required node")?;
    block.append_operation(func::r#return(&outs, loc));

    let region = Region::new();
    region.append_block(block);

    Ok(func::func(
        ctx.context,
        StringAttribute::new(ctx.context, func_name),
        TypeAttribute::new(FunctionType::new(ctx.context, &sig_in, &sig_out).into()),
        region,
        &[],
        loc,
    ))
}

fn compile_node<'c, 'a>(
    idx: NodeIndex,
    block: &'a Block<'c>,
    compile_graph: &mut FuncCompileGraph<'c, 'a>,
    // Might need to be an Either<Info, Vec<Info>>
    infos: &Infos,
    ctx: CompileContext<'c, '_>,
) -> Result<()> {
    if compile_graph.node_weight(idx).unwrap().1.is_some() {
        // This node has already been compiled
        return Ok(());
    }

    let deps = compile_graph.neighbors(idx);
    let dep_edges = compile_graph.edges(idx);
    let (deps, dep_edges): (Vec<_>, Vec<usize>) = deps
        .zip(dep_edges.map(|e| e.weight()))
        .sorted_by_key(|(_, e)| *e)
        .unzip();

    for &dep in &deps {
        compile_node(dep, block, compile_graph, infos, ctx)?;
    }

    use uiua::Primitive::*;
    let data = compile_graph
        .node_weight(idx)
        .context("Node missing from compile graph")?
        .0;
    dbg!(&data);
    let value: Value = match data {
        Data::Arg(i) => block.argument(i).unwrap().into(),
        Data::Out => todo!(),
        Data::Node(Node::Push(value)) => {
            let loc = Location::unknown(ctx.context);
            let arr: uiua::Array<f64> = match value.clone() {
                uiua::Value::Byte(arr) => arr.convert(),
                uiua::Value::Num(arr) => arr,
                _ => bail!("Currently only numbers are supported"),
            };
            let val_type = mk_tensor_type(&ShapeInfo::Known(value.clone()), ctx);
            let elements: Vec<f64> = arr.elements().copied().collect();

            let mut element_values = Vec::new();
            for elem in elements {
                let op = arith::constant(
                    ctx.context,
                    melior::ir::attribute::FloatAttribute::new(ctx.context, ctx.float_type, elem)
                        .into(),
                    loc,
                );
                let val = block.append_operation(op).result(0).unwrap().into();
                element_values.push(val);
            }

            let op = tensor::from_elements(ctx.context, val_type, &element_values, loc).into();
            block.append_operation(op).result(0).unwrap().into()
        }
        Data::Node(Node::Call(func, span)) => {
            let loc = span_to_loc(*span, ctx);
            let func_name = match &func.id {
                uiua::FunctionId::Named(ident) => ident,
                _ => todo!("Non named function"),
            };
            let ref_attr = FlatSymbolRefAttribute::new(ctx.context, func_name);
            let args = deps
                .iter()
                .map(|&idx| compile_graph.node_weight(idx).and_then(|(_, v)| *v))
                .collect::<Option<Vec<_>>>()
                .context("Argument missing from compile graph")?;
            let out_info = infos.get(&idx).context("Did not analyze output")?;
            if out_info.typ != 0 {
                bail!("Only numbers are supported currently");
            }
            let out_type = mk_tensor_type(&out_info.shape, ctx);
            let op = func::call(ctx.context, ref_attr, &args, &[out_type], loc);
            block.append_operation(op).result(0).unwrap().into()
        }
        Data::Node(Node::Prim(Add, span)) => {
            let lhs = deps[0];
            let rhs = deps[1];
            let lhs_info = infos.get(&lhs).context("Did not analyze argument")?;
            let rhs_info = infos.get(&rhs).context("Did not analyze argument")?;
            let out_info = infos.get(&idx).context("Did not analyze output")?;
            if lhs_info.typ != 0 || rhs_info.typ != 0 {
                bail!("Addition is currently only implemented for numbers");
            }

            let lhs_val = compile_graph
                .node_weight(lhs)
                .context("Argument missing from compile graph")?
                .1
                .context("Argument not compiled")?;
            let rhs_val = compile_graph
                .node_weight(rhs)
                .context("Argument missing from compile graph")?
                .1
                .context("Argument not compiled")?;

            // let lhs_type = mk_tensor_type(&lhs_info.shape);
            // let rhs_type = mk_tensor_type(&rhs_info.shape);
            let out_type = mk_tensor_type(&out_info.shape, ctx);

            let loc = span_to_loc(*span, ctx);

            let op = tosa::AddOperationBuilder::new(ctx.context, loc)
                .input_1(lhs_val)
                .input_2(rhs_val)
                .output(out_type)
                .build();
            block.append_operation(op.into()).result(0).unwrap().into()
        }
        Data::Node(node) => todo!(),
    };

    compile_graph.node_weight_mut(idx).unwrap().1 = Some(value);

    Ok(())
}

// fn mk_type<'c>(info: &Info, ctx: CompileContext<'c, '_>) -> Type<'c> {

// }

fn mk_tensor_type<'c>(info: &ShapeInfo, ctx: CompileContext<'c, '_>) -> Type<'c> {
    match info {
        ShapeInfo::Known(value) => {
            let dims = value.shape.iter().map(|len| *len as u64).collect_vec();
            RankedTensorType::new(&dims, ctx.float_type, None)
        }
        ShapeInfo::Ranked(shape) => {
            let dims = shape
                .iter()
                .map(|ax| ax.only_const().map(|len| len as u64).unwrap_or(DYN_AX))
                .collect_vec();
            RankedTensorType::new(&dims, ctx.float_type, None)
        }
        ShapeInfo::Unranked { prefix, suffix } => todo!(),
    }
    .into()
}

fn span_to_loc<'c>(span: usize, ctx: CompileContext<'c, '_>) -> Location<'c> {
    if let Some(sp) = ctx.uiua.get_span(span).code()
        && let uiua::InputSrc::File(path) = sp.src
        && let Some(path) = path.as_os_str().to_str()
    {
        Location::new(
            ctx.context,
            path,
            sp.start.line as usize,
            sp.start.col as usize,
        )
    } else {
        Location::unknown(ctx.context)
    }
}
