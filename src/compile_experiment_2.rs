use anyhow::{bail, Context as AnyhowContext, Result};
use itertools::{Either, Itertools};
use melior::{
    dialect::{
        arith, func,
        ods::{tensor, tosa},
        scf, DialectRegistry,
    },
    ir::{
        attribute::{
            DenseElementsAttribute, FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute,
            StringAttribute, TypeAttribute,
        },
        operation::OperationLike,
        r#type::{FunctionType, RankedTensorType},
        *,
    },
    pass::{self, PassManager},
    utility::register_all_dialects,
    Context,
};
use petgraph::{data::DataMap, graph::NodeIndex, stable_graph::StableGraph};
use std::io::Write;
use uiua::{Node, Purity};

use crate::{
    analyze::{self, analyze_func_graph, axis::Axis, AnalyzedFunc, FuncInfos, FuncLib, ShapeInfo},
    graph::{Data, DataGraph},
};

/// The integer used to indicate a dynamic axis length to MLIR
const DYN_AX: u64 = i64::MAX as u64 + 1;

#[derive(Clone, Copy)]
struct CompileContext<'c, 'u> {
    context: &'c Context,
    index_type: Type<'c>,
    bool_type: Type<'c>,
    float_type: Type<'c>,
    char_type: Type<'c>,
    uiua: &'u uiua::Uiua,
}

// struct CompiledFuncLib<'c> {
//     funcs: Vec<Operation<'c>>,
// }

type FuncCompileGraph<'c, 'a> =
    StableGraph<(Data<'a>, Option<Either<Value<'c, 'a>, Vec<Value<'c, 'a>>>>), usize>;

pub fn compile_test(uiua: &uiua::Uiua) -> Result<()> {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let index_type = Type::index(&context);
    let bool_type = Type::parse(&context, "i1").unwrap();
    let float_type = Type::float64(&context);
    let char_type = Type::parse(&context, "i32").unwrap();
    let ctx = CompileContext {
        context: &context,
        index_type,
        bool_type,
        float_type,
        char_type,
        uiua,
    };

    // let mut module = Module::new(Location::unknown(&context));
    let mut module = Module::parse(ctx.context, include_str!("print_num.mlir"))
        .context("Failed to parse module")?;

    let mut funclib = FuncLib::new();

    let data_graph = DataGraph::from_node(&uiua.asm.root, &uiua.asm)?;

    let infos = analyze_func_graph(&data_graph, &[], &mut funclib, uiua)?;

    let func_id = uiua::FunctionId::Named("main".into());
    let span = uiua.asm.root.span();
    funclib.funcs.push(crate::analyze::AnalyzedFunc::new(
        func_id, data_graph, infos, span,
    ));

    for i in 0..funclib.funcs.len() {
        let func = compile_func(&funclib, i, ctx)?;
        module.body().append_operation(func);
    }

    assert!(module.as_operation().verify());

    println!("before passes");
    println!("{}", module.as_operation());

    let pass_manager = PassManager::new(&context);
    pass_manager.enable_verifier(true);
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    // pass_manager.add_pass(pass::conversion::create_tosa_to_linalg());
    pass_manager.add_pass(pass::conversion::create_scf_to_control_flow()); // needed because to_llvm doesn't include it.
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager.run(&mut module)?;

    println!("after passes");
    println!("{}", module.as_operation());

    let mut f = std::fs::File::create("mlir-test/test.mlir")?;
    write!(f, "{}", module.as_operation())?;

    Ok(())
}

fn compile_func<'c>(
    funclib: &FuncLib,
    idx: usize,
    ctx: CompileContext<'c, '_>,
) -> Result<Operation<'c>> {
    let func = &funclib.funcs[idx];
    let func_name = name_mangle(func)?;

    let loc = func
        .span
        .map(|span| span_to_loc(span, ctx))
        .unwrap_or_else(|| Location::unknown(ctx.context));

    let mut compile_graph: FuncCompileGraph =
        func.graph.graph.map(|_, &data| (data, None), |_, &x| x);

    let mut sig_in = Vec::new();
    let mut arg_types = Vec::new();
    for arg_info in &func.infos.args {
        let elem_type = match arg_info.typ {
            0 => ctx.float_type,
            1 => ctx.char_type,
            _ => unimplemented!(),
        };
        let arg_type = mk_tensor_type(&arg_info.shape, elem_type);
        sig_in.push(arg_type);
        arg_types.push((arg_type, loc));
    }

    let mut sig_out = Vec::new();
    for out_info in &func.infos.outs {
        let elem_type = match out_info.typ {
            0 => ctx.float_type,
            1 => ctx.char_type,
            _ => unimplemented!(),
        };
        let out_type = mk_tensor_type(&out_info.shape, elem_type);
        sig_out.push(out_type);
    }
    if func_name == "main" {
        sig_out.push(ctx.index_type);
    }

    let block = Block::new(&arg_types);

    for root in func.graph.roots(&ctx.uiua.asm) {
        compile_node(root, &block, &mut compile_graph, &func.infos, funclib, ctx)?;
    }

    let outs = if func_name != "main" {
        func.graph
            .stack
            .iter()
            .map(|&idx| {
                compile_graph
                    .node_weight(idx)
                    .and_then(|(_, v)| v.as_ref().and_then(|v| v.as_ref().left()))
                    .copied()
            })
            .collect::<Option<Vec<_>>>()
            .context("Did not compile required node")?
    } else {
        vec![block
            .append_operation(arith::constant(
                ctx.context,
                IntegerAttribute::new(ctx.index_type, 0).into(),
                loc,
            ))
            .result(0)?
            .into()]
    };
    block.append_operation(func::r#return(&outs, loc));

    let region = Region::new();
    region.append_block(block);

    // TODO: Figure out what attributes to put to indicate purity
    // let mut attrs = Vec::new();
    // if func.infos.purity == Purity::Pure {
    //     attrs.push((
    //         Identifier::new(ctx.context, "Pure"),
    //         BoolAttribute::new(ctx.context, true).into(),
    //     ));
    // }

    Ok(func::func(
        ctx.context,
        StringAttribute::new(ctx.context, &func_name),
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
    infos: &FuncInfos,
    funclib: &FuncLib,
    ctx: CompileContext<'c, '_>,
) -> Result<()> {
    if compile_graph.node_weight(idx).unwrap().1.is_some() {
        // This node has already been compiled
        return Ok(());
    }

    let deps = compile_graph.neighbors(idx);
    let dep_edges = compile_graph.edges(idx);
    let (deps, _dep_edges): (Vec<_>, Vec<usize>) = deps
        .zip(dep_edges.map(|e| e.weight()))
        .sorted_by_key(|(_, e)| *e)
        .unzip();

    for &dep in &deps {
        compile_node(dep, block, compile_graph, infos, funclib, ctx)?;
    }

    use uiua::Primitive::*;
    let data = compile_graph
        .node_weight(idx)
        .expect("Node missing from compile graph")
        .0;
    dbg!(&data);
    let value = match data {
        Data::Arg(i) => Either::Left(block.argument(i)?.into()),
        Data::Out => todo!(),
        Data::Node(Node::Push(value)) => Either::Left(constant(value, block, ctx)?),
        Data::Node(Node::Call(_func, span)) => {
            let loc = span_to_loc(*span, ctx);
            let analyzed_func = &funclib.funcs[infos.map.get(&idx).unwrap().subfunc_idxs[0]];
            let func_name = name_mangle(analyzed_func)?;
            let ref_attr = FlatSymbolRefAttribute::new(ctx.context, &func_name);
            let args = deps
                .iter()
                .map(|&idx| {
                    compile_graph
                        .node_weight(idx)
                        .and_then(|(_, v)| v.as_ref().and_then(|v| v.as_ref().left()))
                        .copied()
                })
                .collect::<Option<Vec<_>>>()
                .expect("Argument missing from compile graph");
            let out_info = infos.map.get(&idx).context("Did not analyze output")?;
            let elem_type = match out_info.typ {
                0 => ctx.float_type,
                1 => ctx.char_type,
                _ => unimplemented!(),
            };
            let out_type = mk_tensor_type(&out_info.shape, elem_type);
            let op = func::call(ctx.context, ref_attr, &args, &[out_type], loc);
            Either::Left(block.append_operation(op).result(0)?.into())
        }
        Data::Node(Node::Prim(Add, span)) => {
            let lhs = deps[0];
            let rhs = deps[1];
            let lhs_info = infos.map.get(&lhs).expect("Did not analyze argument");
            let rhs_info = infos.map.get(&rhs).expect("Did not analyze argument");
            let out_info = infos.map.get(&idx).expect("Did not analyze output");
            if lhs_info.typ != 0 || rhs_info.typ != 0 {
                bail!("Addition is currently only implemented for numbers");
            }

            let lhs_val = *compile_graph
                .node_weight(lhs)
                .expect("Argument missing from compile graph")
                .1
                .as_ref()
                .and_then(|v| v.as_ref().left())
                .expect("Argument not compiled");
            let rhs_val = *compile_graph
                .node_weight(rhs)
                .expect("Argument missing from compile graph")
                .1
                .as_ref()
                .and_then(|v| v.as_ref().left())
                .expect("Argument not compiled");

            let out_type = mk_tensor_type(&out_info.shape, ctx.float_type);

            let loc = span_to_loc(*span, ctx);

            let op = tosa::AddOperationBuilder::new(ctx.context, loc)
                .input_1(lhs_val)
                .input_2(rhs_val)
                .output(out_type)
                .build();
            Either::Left(block.append_operation(op.into()).result(0)?.into())
        }
        Data::Node(Node::Prim(Sys(uiua::SysOp::Print), span)) => {
            let loc = span_to_loc(*span, ctx);
            let arg_info = infos.map.get(&deps[0]).expect("Did not analyze argument");
            let (elem_type, print_func) = match arg_info.typ {
                0 => (ctx.float_type, "print_f64"),
                1 => (ctx.char_type, "print_i32"),
                _ => unimplemented!(),
            };
            let len = match &arg_info.shape {
                ShapeInfo::Known(val) => val.shape.elements() as u64,
                ShapeInfo::Ranked(shape) => shape
                    .iter()
                    .product::<Axis>()
                    .only_const()
                    .map(|ax| ax as u64)
                    .unwrap_or(DYN_AX),
                ShapeInfo::Unranked { .. } => DYN_AX,
            };
            let out_type = RankedTensorType::new(&[len], elem_type, None);

            let arg_val = *compile_graph
                .node_weight(deps[0])
                .expect("Argument missing from compile graph")
                .1
                .as_ref()
                .and_then(|v| v.as_ref().left())
                .expect("Argument not compiled");

            let flat_shape_val: Value = block
                .append_operation(arith::constant(
                    ctx.context,
                    DenseElementsAttribute::new(
                        RankedTensorType::new(&[1], ctx.index_type, None).into(),
                        &[IntegerAttribute::new(ctx.index_type, len as i64).into()],
                    )?
                    .into(),
                    loc,
                ))
                .result(0)?
                .into();

            let deshape_op =
                tensor::reshape(ctx.context, out_type.into(), arg_val, flat_shape_val, loc);

            let deshaped_val: Value = block.append_operation(deshape_op.into()).result(0)?.into();

            let zero_val: Value = block
                .append_operation(arith::constant(
                    ctx.context,
                    IntegerAttribute::new(ctx.index_type, 0).into(),
                    loc,
                ))
                .result(0)?
                .into();
            let one_val: Value = block
                .append_operation(arith::constant(
                    ctx.context,
                    IntegerAttribute::new(ctx.index_type, 1).into(),
                    loc,
                ))
                .result(0)?
                .into();

            let len_op = tensor::dim(ctx.context, ctx.index_type, deshaped_val, zero_val, loc);
            let len_val: Value = block.append_operation(len_op.into()).result(0)?.into();

            let for_op = scf::r#for(
                zero_val,
                len_val,
                one_val,
                {
                    let for_block = Block::new(&[(ctx.index_type, loc)]);
                    let idx_val: Value = for_block.argument(0)?.into();
                    let get_op =
                        tensor::extract(ctx.context, elem_type, deshaped_val, &[idx_val], loc);
                    let cur_val: Value =
                        for_block.append_operation(get_op.into()).result(0)?.into();
                    let printf_op = func::call(
                        ctx.context,
                        FlatSymbolRefAttribute::new(ctx.context, print_func),
                        &[cur_val],
                        &[],
                        loc,
                    );
                    for_block.append_operation(printf_op);
                    for_block.append_operation(scf::r#yield(&[], loc));
                    let region = Region::new();
                    region.append_block(for_block);
                    region
                },
                loc,
            );

            block.append_operation(for_op);

            let final_print_op = func::call(
                ctx.context,
                FlatSymbolRefAttribute::new(ctx.context, "print_ln"),
                &[],
                &[],
                loc,
            );
            block.append_operation(final_print_op);

            Either::Right(Vec::with_capacity(0))
        }
        Data::Node(Node::Mod(Rows, _funcs, span)) => {
            let loc = span_to_loc(*span, ctx);

            let lhs = deps[0];
            let rhs = deps[1];
            let lhs_info = infos.map.get(&lhs).expect("Did not analyze argument");
            let rhs_info = infos.map.get(&rhs).expect("Did not analyze argument");
            let out_info = infos.map.get(&idx).expect("Did not analyze output");

            let out_elem_type = match out_info.typ {
                0 => ctx.float_type,
                1 => ctx.char_type,
                _ => unimplemented!(),
            };
            let out_type = mk_tensor_type(&out_info.shape, out_elem_type);

            let (subfunc_graph, subfunc_infos) = &infos.subfuncs[out_info.subfunc_idxs[0]];

            // ---

            let mut compile_graph: FuncCompileGraph =
                subfunc_graph.graph.map(|_, &data| (data, None), |_, &x| x);

            // ---

            todo!()
        }
        Data::Node(node) => todo!(),
    };

    compile_graph.node_weight_mut(idx).unwrap().1 = Some(value);

    Ok(())
}

fn mk_tensor_type<'c>(info: &ShapeInfo, typ: Type<'c>) -> Type<'c> {
    match info {
        ShapeInfo::Known(value) => {
            let dims = value.shape.iter().map(|len| *len as u64).collect_vec();
            RankedTensorType::new(&dims, typ, None).into()
        }
        ShapeInfo::Ranked(shape) => {
            let dims = shape
                .iter()
                .map(|ax| ax.only_const().map(|len| len as u64).unwrap_or(DYN_AX))
                .collect_vec();
            RankedTensorType::new(&dims, typ, None).into()
        }
        ShapeInfo::Unranked { prefix, suffix } => todo!(),
    }
}

fn name_mangle(func: &AnalyzedFunc) -> Result<String> {
    let uiua::FunctionId::Named(base_name) = &func.id else {
        bail!("Attempted to mangle non-named function");
    };

    if base_name == "main" {
        return Ok(base_name.into());
    }

    let name_suffix = func
        .infos
        .args
        .iter()
        .map(|info| {
            format!(
                "{}t{}",
                match &info.shape {
                    ShapeInfo::Known(val) => val.shape.iter().map(|ax| ax.to_string()).join("x"),
                    ShapeInfo::Ranked(shape) => shape
                        .iter()
                        .map(|ax| {
                            ax.only_const()
                                .map(|ax| ax.to_string())
                                .unwrap_or_else(|| "X".to_owned())
                        })
                        .join("x"),
                    ShapeInfo::Unranked { .. } => "U".to_owned(),
                },
                info.typ
            )
        })
        .join("_");
    Ok(format!("__{base_name}__{name_suffix}"))
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

fn constant<'c, 'a>(
    value: &uiua::Value,
    block: &'a Block<'c>,
    ctx: CompileContext<'c, '_>,
) -> Result<Value<'c, 'a>> {
    let loc = Location::unknown(ctx.context);
    let attr = if let Some(num_arr) = value
        .as_num_array()
        .cloned()
        .or_else(|| value.as_byte_array().cloned().map(uiua::Array::convert))
    {
        let tensor_type = mk_tensor_type(&ShapeInfo::Known(value.clone()), ctx.float_type);

        let elem_attrs = num_arr
            .elements()
            .map(|&elem| FloatAttribute::new(ctx.context, ctx.float_type, elem).into())
            .collect_vec();
        DenseElementsAttribute::new(tensor_type, &elem_attrs)?
    } else if let Some(char_arr) = value.as_char_array() {
        let tensor_type = mk_tensor_type(&ShapeInfo::Known(value.clone()), ctx.char_type);

        let elem_attrs = char_arr
            .elements()
            .map(|&elem| IntegerAttribute::new(ctx.char_type, elem as i64).into())
            .collect_vec();
        DenseElementsAttribute::new(tensor_type, &elem_attrs)?
    } else {
        unimplemented!()
    };

    let op = arith::constant(ctx.context, attr.into(), loc);
    Ok(block.append_operation(op).result(0)?.into())
}
