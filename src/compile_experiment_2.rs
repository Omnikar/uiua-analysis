use anyhow::{bail, Context as _, Result};
use itertools::Itertools;
use melior::{
    dialect::{
        arith, func,
        ods::{scf, tensor, tosa},
        DialectRegistry,
    },
    ir::{
        attribute::{
            DenseElementsAttribute, DenseI32ArrayAttribute, DenseI64ArrayAttribute,
            FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute, StringAttribute,
            TypeAttribute,
        },
        operation::{OperationBuilder, OperationLike},
        r#type::{FunctionType, RankedTensorType},
        *,
    },
    pass,
    utility::register_all_dialects,
    Context,
};
use petgraph::{data::DataMap, graph::NodeIndex, stable_graph::StableGraph};
use std::io::Write;
use uiua::{Node, Purity};

use crate::{
    analyze::{
        analyze_func_graph, axis::Axis, AnalyzedFunc, FuncInfos, FuncLib, InfoMap, ShapeInfo,
    },
    graph::{Data, DataGraph, Stack},
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

// Using a `SmallVec` causes lifetime issues for some reason
type FuncCompileGraph<'c, 'a, 'u> =
    StableGraph<(Data<'u>, Option<Vec<Value<'c, 'a>>>), (usize, usize)>;

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
    funclib
        .funcs
        .push(AnalyzedFunc::new(func_id, data_graph, infos, span));

    for i in 0..funclib.funcs.len() {
        let func = compile_func(&funclib, i, ctx)?;
        module.body().append_operation(func);
    }

    // println!("before verify");
    println!("{}", module.as_operation());

    assert!(module.as_operation().verify());

    // println!("before passes");
    // println!("{}", module.as_operation());

    // let pass_manager = pass::PassManager::new(&context);
    // pass_manager.enable_verifier(true);
    // pass_manager.add_pass(pass::transform::create_canonicalizer());
    // // pass_manager.add_pass(pass::conversion::create_tosa_to_linalg());
    // pass_manager.add_pass(pass::conversion::create_scf_to_control_flow()); // needed because to_llvm doesn't include it.
    // pass_manager.add_pass(pass::conversion::create_to_llvm());
    // pass_manager.run(&mut module)?;

    // println!("after passes");
    // println!("{}", module.as_operation());

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
        compile_node(
            root,
            &block,
            &mut compile_graph,
            &func.infos,
            &func.infos.map,
            funclib,
            ctx,
        )?;
    }

    let outs = if func_name != "main" {
        func.graph
            .stack
            .iter()
            .map(|&(idx, out_i)| Some(compile_graph.node_weight(idx)?.1.as_ref()?[out_i]))
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

fn compile_node<'c, 'a, 'u>(
    idx: NodeIndex,
    block: &'a Block<'c>,
    compile_graph: &mut FuncCompileGraph<'c, 'a, 'u>,
    infos: &FuncInfos<'u>,
    info_map: &InfoMap,
    funclib: &FuncLib<'u>,
    ctx: CompileContext<'c, 'u>,
) -> Result<()> {
    if compile_graph.node_weight(idx).unwrap().1.is_some() {
        // This node has already been compiled
        return Ok(());
    }

    let deps = compile_graph.neighbors(idx);
    let dep_edges = compile_graph.edges(idx);
    let deps: Stack = deps
        .zip(dep_edges.map(|e| *e.weight()))
        .sorted_by_key(|(_, (_, in_i))| *in_i)
        .map(|(idx, (out_i, _))| (idx, out_i))
        .collect();

    for &(dep, _) in &deps {
        compile_node(dep, block, compile_graph, infos, info_map, funclib, ctx)?;
    }

    use uiua::Primitive::*;
    let data = compile_graph
        .node_weight(idx)
        .expect("Node missing from compile graph")
        .0;
    dbg!(&data);
    let value = match data {
        Data::Arg(i) => vec![block.argument(i)?.into()],
        Data::Node(Node::Push(value)) => vec![constant(value, block, ctx)?],
        Data::Node(Node::Call(_func, span)) => {
            let loc = span_to_loc(*span, ctx);
            let node_info = info_map.get(&idx).expect("Did not analyze output");
            let analyzed_func = &funclib.funcs[node_info.subfunc_idxs[0]];
            let func_name = name_mangle(analyzed_func)?;
            let ref_attr = FlatSymbolRefAttribute::new(ctx.context, &func_name);
            let args = deps
                .iter()
                .map(|&(idx, out_i)| {
                    compile_graph
                        .node_weight(idx)
                        .and_then(|(_, v)| v.as_ref()?.get(out_i).copied())
                })
                .collect::<Option<Vec<_>>>()
                .expect("Argument missing from compile graph");

            let out_types = node_info
                .vals
                .iter()
                .map(|out_info| {
                    let elem_type = match out_info.typ {
                        0 => ctx.float_type,
                        1 => ctx.char_type,
                        _ => unimplemented!(),
                    };
                    mk_tensor_type(&out_info.shape, elem_type)
                })
                .collect_vec();

            let op = func::call(ctx.context, ref_attr, &args, &out_types, loc);
            let op_ref = block.append_operation(op);
            (0..out_types.len())
                .map(|i| op_ref.result(i).map(Into::into).map_err(Into::into))
                .collect::<Result<_>>()?
        }
        Data::Node(Node::Prim(Add, span)) => {
            let (lhs_idx, lhs_out_i) = deps[0];
            let (rhs_idx, rhs_out_i) = deps[1];
            let lhs_info = &info_map
                .get(&lhs_idx)
                .expect("Did not analyze argument")
                .vals[lhs_out_i];
            let rhs_info = &info_map
                .get(&rhs_idx)
                .expect("Did not analyze argument")
                .vals[rhs_out_i];
            let out_info = &info_map.get(&idx).expect("Did not analyze output").vals[0];
            if lhs_info.typ != 0 || rhs_info.typ != 0 {
                bail!("Addition is currently only implemented for numbers");
            }

            let lhs_val = *compile_graph
                .node_weight(lhs_idx)
                .expect("Argument missing from compile graph")
                .1
                .as_ref()
                .and_then(|v| v.get(lhs_out_i))
                .expect("Argument not compiled");
            let rhs_val = *compile_graph
                .node_weight(rhs_idx)
                .expect("Argument missing from compile graph")
                .1
                .as_ref()
                .and_then(|v| v.get(rhs_out_i))
                .expect("Argument not compiled");

            let out_type = mk_tensor_type(&out_info.shape, ctx.float_type);

            let loc = span_to_loc(*span, ctx);

            let op = tosa::AddOperationBuilder::new(ctx.context, loc)
                .input_1(lhs_val)
                .input_2(rhs_val)
                .output(out_type)
                .build();
            vec![block.append_operation(op.into()).result(0)?.into()]
        }
        Data::Node(Node::Prim(Sys(uiua::SysOp::Print), span)) => {
            let loc = span_to_loc(*span, ctx);
            let (arg_idx, arg_out_i) = deps[0];
            let arg_info = &info_map
                .get(&arg_idx)
                .expect("Did not analyze argument")
                .vals[arg_out_i];
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
                .node_weight(arg_idx)
                .expect("Argument missing from compile graph")
                .1
                .as_ref()
                .and_then(|v| v.get(arg_out_i))
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
                ctx.context,
                &[],
                zero_val,
                len_val,
                one_val,
                &[],
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
                    for_block.append_operation(scf::r#yield(ctx.context, &[], loc).into());
                    let region = Region::new();
                    region.append_block(for_block);
                    region
                },
                loc,
            );

            block.append_operation(for_op.into());

            let final_print_op = func::call(
                ctx.context,
                FlatSymbolRefAttribute::new(ctx.context, "print_ln"),
                &[],
                &[],
                loc,
            );
            block.append_operation(final_print_op);

            vec![]
        }
        Data::Node(Node::Mod(Rows, _funcs, span)) => {
            let loc = span_to_loc(*span, ctx);

            let dep_vals = deps
                .iter()
                .map(|&(idx, out_i)| {
                    compile_graph.node_weight(idx).unwrap().1.as_ref().unwrap()[out_i]
                })
                .collect_vec();
            let dep_infos = deps
                .iter()
                .map(|&(idx, out_i)| &info_map.get(&idx).unwrap().vals[out_i])
                .collect_vec();
            let node_info = info_map.get(&idx).unwrap();

            let (subfunc_graph, subfunc_info_map) = &infos.subfuncs[node_info.subfunc_idxs[0]];
            dbg!(&subfunc_graph, &subfunc_info_map);

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

            let len_op = tensor::dim(ctx.context, ctx.index_type, dep_vals[0], zero_val, loc);
            let len_val: Value = block.append_operation(len_op.into()).result(0)?.into();

            // let (out_types, out_inits): (Vec<_>, Vec<_>) = node_info
            //     .vals
            //     .iter()
            let (out_types, out_inits): (Vec<_>, Vec<_>) = subfunc_graph
                .stack
                .iter()
                .map(|(out_idx, out_i)| &subfunc_info_map.get(out_idx).unwrap().vals[*out_i])
                .map(|val_info| {
                    let elem_type = match val_info.typ {
                        0 => ctx.float_type,
                        1 => ctx.char_type,
                        _ => unimplemented!(),
                    };
                    // let tensor_type = mk_tensor_type(&val_info.shape, elem_type);
                    let dims = dims_from_shape_info(&val_info.shape);
                    // dbg!(&val_info, &dims);
                    // let mut dyn_axes = Vec::new();
                    // for (dim_i, dim) in dims.into_iter().enumerate() {
                    //     if dim == DYN_AX {
                    //         let dim_op = tensor::dim(ctx.context,
                    //             ctx.index_type, )
                    //     }
                    // }

                    let mut out_dims = dims_from_shape_info(&dep_infos[0].shape);
                    out_dims.truncate(1);
                    out_dims.extend_from_slice(&dims);
                    let out_type: melior::ir::Type =
                        RankedTensorType::new(&out_dims, elem_type, None).into();

                    let empty_op = tensor::empty(ctx.context, out_type, &[], loc);
                    let init = block
                        .append_operation(empty_op.into())
                        .result(0)
                        .map(Value::from)
                        .map_err(Into::into);

                    (out_type, init)
                })
                .unzip();
            // .collect::<Result<Vec<_>>>()?;
            let out_inits = out_inits.into_iter().collect::<Result<Vec<_>>>()?;

            let mut for_block_args = out_inits
                .iter()
                .map(|val| (val.r#type(), loc))
                .collect_vec();
            for_block_args.insert(0, (ctx.index_type, loc));
            // let for_block = Block::new(&[(ctx.index_type, loc)]);
            let for_block = Block::new(&for_block_args);
            let idx_val: Value = for_block.argument(0)?.into();
            let accs: Vec<Value> = (1..=out_inits.len())
                .map(|i| Ok(for_block.argument(i).map(Into::into)?))
                .collect::<Result<_>>()?;

            let extracted: Vec<Value> = (0..deps.len())
                .map(|arg_i| {
                    let dep_val = dep_vals[arg_i];
                    let dep_info = dep_infos[arg_i];
                    let elem_type = match dep_info.typ {
                        0 => ctx.float_type,
                        1 => ctx.char_type,
                        _ => unimplemented!(),
                    };
                    let inner_dims = dims_from_shape_info(&dep_info.shape);
                    let inner_dims = &inner_dims[1..];
                    let inner_type = RankedTensorType::new(inner_dims, elem_type, None).into();

                    let mut static_offsets = vec![0; inner_dims.len()];
                    static_offsets.insert(0, DYN_AX as i64);

                    let mut sizes = Vec::new();
                    let mut static_sizes = vec![1];
                    for (i, &dim) in inner_dims.iter().enumerate() {
                        static_sizes.push(dim as i64);
                        if dim == DYN_AX {
                            let dim_i: Value = for_block
                                .append_operation(arith::constant(
                                    ctx.context,
                                    IntegerAttribute::new(ctx.index_type, i as i64 + 1).into(),
                                    loc,
                                ))
                                .result(0)?
                                .into();
                            let len: Value = for_block
                                .append_operation(
                                    tensor::dim(ctx.context, ctx.index_type, dep_val, dim_i, loc)
                                        .into(),
                                )
                                .result(0)?
                                .into();
                            sizes.push(len);
                        }
                    }

                    // FIXME: For some reason melior was outputting an `operandSegmentSizes` field of all 0s, so I had to specify it myself
                    // let get_op = tensor::extract_slice(
                    //     ctx.context,
                    //     inner_type,
                    //     dep_val,
                    //     &[idx_val],
                    //     &sizes,
                    //     &[],
                    //     DenseI64ArrayAttribute::new(ctx.context, &static_offsets).into(),
                    //     DenseI64ArrayAttribute::new(ctx.context, &static_sizes).into(),
                    //     DenseI64ArrayAttribute::new(
                    //         ctx.context,
                    //         &vec![1; inner_dims.len() + 1],
                    //     )
                    //     .into(),
                    //     loc,
                    // );
                    let get_op = OperationBuilder::new("tensor.extract_slice", loc)
                        .add_results(&[inner_type])
                        .add_operands(&[dep_val, idx_val])
                        .add_operands(&sizes)
                        .add_attributes(&[
                            (
                                Identifier::new(ctx.context, "static_offsets"),
                                DenseI64ArrayAttribute::new(ctx.context, &static_offsets).into(),
                            ),
                            (
                                Identifier::new(ctx.context, "static_sizes"),
                                DenseI64ArrayAttribute::new(ctx.context, &static_sizes).into(),
                            ),
                            (
                                Identifier::new(ctx.context, "static_strides"),
                                DenseI64ArrayAttribute::new(
                                    ctx.context,
                                    &vec![1; inner_dims.len() + 1],
                                )
                                .into(),
                            ),
                            (
                                Identifier::new(ctx.context, "operandSegmentSizes"),
                                DenseI32ArrayAttribute::new(
                                    ctx.context,
                                    &[1, 1, sizes.len() as i32, 0],
                                )
                                .into(),
                            ),
                        ])
                        .build()?;
                    for_block
                        .append_operation(get_op)
                        .result(0)
                        .map_err(Into::into)
                        .map(Into::into)
                })
                .collect::<Result<_>>()?;

            let mut compile_graph: FuncCompileGraph = subfunc_graph.graph.map(
                |_, &data| {
                    (
                        data,
                        match data {
                            Data::Arg(i) => Some(vec![extracted[i]]),
                            _ => None,
                        },
                    )
                },
                |_, &x| x,
            );

            for root in subfunc_graph.roots(&ctx.uiua.asm) {
                compile_node(
                    root,
                    &for_block,
                    &mut compile_graph,
                    infos,
                    subfunc_info_map,
                    funclib,
                    ctx,
                )?;
            }

            let mut yield_vals = Vec::new();
            for (&(out_idx, out_i), acc) in subfunc_graph.stack.iter().zip(accs) {
                let out_val = compile_graph
                    .node_weight(out_idx)
                    .unwrap()
                    .1
                    .as_ref()
                    .unwrap()[out_i];

                let out_info = &subfunc_info_map.get(&out_idx).unwrap().vals[out_i];

                // let elem_type = match out_info.typ {
                //     0 => ctx.float_type,
                //     1 => ctx.char_type,
                //     _ => unimplemented!(),
                // };
                let out_dims = dims_from_shape_info(&out_info.shape);

                let mut static_offsets = vec![0; out_dims.len()];
                static_offsets.insert(0, DYN_AX as i64);

                let mut sizes = Vec::new();
                let mut static_sizes = vec![1];
                for (i, &dim) in out_dims.iter().enumerate() {
                    static_sizes.push(dim as i64);
                    if dim == DYN_AX {
                        let dim_i: Value = for_block
                            .append_operation(arith::constant(
                                ctx.context,
                                IntegerAttribute::new(ctx.index_type, i as i64 + 1).into(),
                                loc,
                            ))
                            .result(0)?
                            .into();
                        let len: Value = for_block
                            .append_operation(
                                tensor::dim(ctx.context, ctx.index_type, out_val, dim_i, loc)
                                    .into(),
                            )
                            .result(0)?
                            .into();
                        sizes.push(len);
                    }
                }

                // let insert_op = tensor::insert_slice(
                //     ctx.context,
                //     acc.r#type(),
                //     out_val,
                //     acc,
                //     &[idx_val],
                //     &sizes,
                //     &[],
                //     DenseI64ArrayAttribute::new(ctx.context, &static_offsets).into(),
                //     DenseI64ArrayAttribute::new(ctx.context, &static_sizes).into(),
                //     DenseI64ArrayAttribute::new(ctx.context, &vec![1; out_dims.len() + 1]).into(),
                //     loc,
                // );
                let insert_op = OperationBuilder::new("tensor.insert_slice", loc)
                    .add_results(&[acc.r#type()])
                    .add_operands(&[out_val, acc, idx_val])
                    .add_operands(&sizes)
                    .add_attributes(&[
                        (
                            Identifier::new(ctx.context, "static_offsets"),
                            DenseI64ArrayAttribute::new(ctx.context, &static_offsets).into(),
                        ),
                        (
                            Identifier::new(ctx.context, "static_sizes"),
                            DenseI64ArrayAttribute::new(ctx.context, &static_sizes).into(),
                        ),
                        (
                            Identifier::new(ctx.context, "static_strides"),
                            DenseI64ArrayAttribute::new(ctx.context, &vec![1; out_dims.len() + 1])
                                .into(),
                        ),
                        (
                            Identifier::new(ctx.context, "operandSegmentSizes"),
                            DenseI32ArrayAttribute::new(
                                ctx.context,
                                &[1, 1, 1, sizes.len() as i32, 0],
                            )
                            .into(),
                        ),
                    ])
                    .build()?;
                let out_acc: Value = for_block.append_operation(insert_op).result(0)?.into();
                yield_vals.push(out_acc);
            }

            for_block.append_operation(scf::r#yield(ctx.context, &yield_vals, loc).into());

            let for_region = Region::new();
            for_region.append_block(for_block);

            // let for_op = scf::r#for(
            //     ctx.context,
            //     &out_types,
            //     zero_val,
            //     len_val,
            //     one_val,
            //     &out_inits,
            //     for_region,
            //     loc,
            // );
            let for_op = OperationBuilder::new("scf.for", loc)
                .add_results(&out_types)
                .add_operands(&[zero_val, len_val, one_val])
                .add_operands(&out_inits)
                .add_attributes(&[(
                    Identifier::new(ctx.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(ctx.context, &[1, 1, 1, out_inits.len() as i32])
                        .into(),
                )])
                .add_regions([for_region])
                .build()?;

            let op_ref = block.append_operation(for_op);

            // (0..out_inits.len())
            //     .map(|i| op_ref.result(i).map(Into::into))
            //     .collect::<Result<Vec<_>, _>>()?
            (0..out_inits.len())
                .map(|i| op_ref.result(i).unwrap().into())
                .collect()
        }
        Data::Node(node) => todo!(),
    };

    compile_graph.node_weight_mut(idx).unwrap().1 = Some(value);

    Ok(())
}

fn dims_from_shape_info(info: &ShapeInfo) -> Vec<u64> {
    match info {
        ShapeInfo::Known(value) => value.shape.iter().map(|len| *len as u64).collect_vec(),
        ShapeInfo::Ranked(shape) => shape
            .iter()
            .map(|ax| ax.only_const().map(|len| len as u64).unwrap_or(DYN_AX))
            .collect_vec(),
        ShapeInfo::Unranked { prefix, suffix } => todo!(),
    }
}

fn mk_tensor_type<'c>(info: &ShapeInfo, typ: Type<'c>) -> Type<'c> {
    let dims = dims_from_shape_info(info);
    RankedTensorType::new(&dims, typ, None).into()
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
    Ok(format!("_{base_name}__{name_suffix}"))
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
