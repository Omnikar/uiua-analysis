use anyhow::{bail, Context as _, Result};
use itertools::Itertools;
use melior::{
    dialect::{
        func,
        ods::{arith, scf, tensor, tosa},
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
use uiua::{Node, Purity, SysOp};

use crate::{
    analyze::{
        analyze_func_graph, axis::Axis, AnalyzedFunc, FuncInfos, FuncLib, InfoMap, ShapeInfo,
        ValInfo,
    },
    graph::{Data, DataGraph, Stack},
    pre_compile::{self, prepare_graph, Cast, CompNode, CompType, Impl, Op, PreCompileGraph},
};

/// The integer used to indicate a dynamic axis length to MLIR
const DYN_AX: u64 = i64::MAX as u64 + 1;

type FuncCompileGraph<'c, 'a, 'u> =
    StableGraph<(CompNode<'u>, Option<Vec<Value<'c, 'a>>>), (usize, usize)>;

#[derive(Clone, Copy)]
struct CompileContext<'c, 'u> {
    context: &'c Context,
    index_type: Type<'c>,
    bool_type: Type<'c>,
    int_types: [Type<'c>; 4],
    float_types: [Type<'c>; 2],
    uiua: &'u uiua::Uiua,
}

pub fn compile_test(uiua: &uiua::Uiua) -> Result<()> {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let ctx = CompileContext {
        context: &context,
        index_type: Type::index(&context),
        bool_type: Type::parse(&context, "i1").unwrap(),
        int_types: [
            Type::parse(&context, "i8").unwrap(),
            Type::parse(&context, "i16").unwrap(),
            Type::parse(&context, "i32").unwrap(),
            Type::parse(&context, "i64").unwrap(),
        ],
        float_types: [Type::float32(&context), Type::float64(&context)],
        uiua,
    };

    let module = Module::parse(ctx.context, include_str!("print_num.mlir"))
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

    println!("{}", module.as_operation());

    assert!(module.as_operation().verify());

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

    let pre_compile_graph = prepare_graph(&func.graph, &func.infos.map, ctx.uiua);
    let mut compile_graph: FuncCompileGraph = pre_compile_graph
        .graph
        .map_owned(|_, node| (node, None), |_, x| x);

    let mut sig_in = Vec::new();
    let mut arg_types = Vec::new();
    for arg_info in &func.infos.args {
        let arg_type = mk_type(arg_info, ctx);
        sig_in.push(arg_type);
        arg_types.push((arg_type, loc));
    }

    let mut sig_out = Vec::new();
    for out_info in &func.infos.outs {
        let out_type = mk_type(out_info, ctx);
        sig_out.push(out_type);
    }
    if func_name == "main" {
        sig_out.clear();
        sig_out.push(ctx.index_type);
    }

    let block = Block::new(&arg_types);

    let idxs = compile_graph.node_indices().collect_vec();

    let mut fctx = FuncCompileContext {
        compile_graph: &mut compile_graph,
        func_infos: &func.infos,
        funclib,
    };

    // TODO: Find a way to transfer the roots over and use that instead
    for idx in idxs {
        compile_node(idx, &block, &mut fctx, ctx)?;
    }

    let outs = if func_name != "main" {
        pre_compile_graph
            .stack
            .iter()
            .map(|&(idx, out_i)| Some(compile_graph.node_weight(idx)?.1.as_ref()?[out_i]))
            .collect::<Option<Vec<_>>>()
            .context("Did not compile required node")?
    } else {
        vec![block
            .append_operation(
                arith::constant(
                    ctx.context,
                    ctx.index_type,
                    IntegerAttribute::new(ctx.index_type, 0).into(),
                    loc,
                )
                .into(),
            )
            .result(0)?
            .into()]
    };
    block.append_operation(func::r#return(&outs, loc));

    let region = Region::new();
    region.append_block(block);

    // TODO: Figure out what attributes to put to indicate purity

    Ok(func::func(
        ctx.context,
        StringAttribute::new(ctx.context, &func_name),
        TypeAttribute::new(FunctionType::new(ctx.context, &sig_in, &sig_out).into()),
        region,
        &[],
        loc,
    ))
}

struct FuncCompileContext<'c, 'a, 'u, 'cg, 'fi, 'fl> {
    compile_graph: &'cg mut FuncCompileGraph<'c, 'a, 'u>,
    func_infos: &'fi FuncInfos<'u>,
    funclib: &'fl FuncLib<'u>,
}

fn compile_node<'c, 'a, 'u>(
    idx: NodeIndex,
    block: &'a Block<'c>,
    // info_map: &InfoMap,
    fctx: &mut FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<()> {
    if fctx.compile_graph.node_weight(idx).unwrap().1.is_some() {
        // This node has already been compiled
        return Ok(());
    }

    let deps = fctx.compile_graph.neighbors(idx);
    let dep_edges = fctx.compile_graph.edges(idx);
    let deps: Stack = deps
        .zip(dep_edges.map(|e| *e.weight()))
        .sorted_by_key(|(_, (_, in_i))| *in_i)
        .map(|(idx, (out_i, _))| (idx, out_i))
        .collect();

    for &(dep, _) in &deps {
        compile_node(dep, block, /*info_map,*/ fctx, ctx)?;
    }

    let comp_node = &fctx
        .compile_graph
        .node_weight(idx)
        .expect("Node missing from compile graph")
        .0;

    use uiua::Primitive::*;
    let value = match comp_node.op {
        Op::Data(Data::Arg(i)) => vec![block.argument(i)?.into()],
        Op::Data(Data::Node(Node::Push(value))) => vec![constant(value, block, ctx)?],
        Op::Data(Data::Node(Node::Call(_func, span))) => call(idx, &deps, *span, block, fctx, ctx)?,

        Op::Impl(Impl::Cast(cast), span) => {
            vec![upcast_num(cast, idx, &deps, span, block, fctx, ctx)?]
        }

        Op::Data(Data::Node(Node::Prim(Add, span))) => {
            vec![add(idx, &deps, *span, block, fctx, ctx)?]
        }

        Op::Data(Data::Node(Node::Prim(Sys(SysOp::Print), span))) => {
            print(&deps, *span, block, fctx, ctx)?;
            Vec::new()
        }
        _ => todo!(),
    };

    fctx.compile_graph.node_weight_mut(idx).unwrap().1 = Some(value);

    Ok(())
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
                crate::pre_compile::CompType::from_info(info),
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

fn mk_elem_type<'c>(comp_type: &CompType, ctx: CompileContext<'c, '_>) -> Type<'c> {
    match comp_type {
        CompType::Int(_, i) => ctx.int_types[*i as usize],
        CompType::Float(d) => ctx.float_types[*d as usize],
        CompType::Bool => ctx.bool_type,
        // TODO: perchance?
        CompType::Char => ctx.int_types[2],
    }
}
fn mk_tensor_type<'c>(info: &ShapeInfo, elem_type: Type<'c>) -> Type<'c> {
    let dims = dims_from_shape_info(info);
    RankedTensorType::new(&dims, elem_type, None).into()
}
fn mk_type_from_comp_shape<'c>(
    comp_type: &CompType,
    shape: &ShapeInfo,
    ctx: CompileContext<'c, '_>,
) -> Type<'c> {
    let elem_type = mk_elem_type(comp_type, ctx);
    mk_tensor_type(shape, elem_type)
}
fn mk_type<'c>(info: &ValInfo, ctx: CompileContext<'c, '_>) -> Type<'c> {
    let comp_type = CompType::from_info(info);
    mk_type_from_comp_shape(&comp_type, &info.shape, ctx)
}

/// Returns `ValInfo`s and `Value`s for the dependencies at the given indices
fn get_deps<'c, 'a, 'cg>(
    deps: &[(NodeIndex, usize)],
    compile_graph: &'cg FuncCompileGraph<'c, 'a, '_>,
) -> (Vec<&'cg ValInfo>, Vec<Value<'c, 'a>>) {
    deps.iter()
        .map(|&(dep_idx, out_i)| {
            let node = &compile_graph.node_weight(dep_idx).unwrap();
            (
                &node.0.info.vals[out_i],
                node.1.as_ref().expect("Argument not compiled")[out_i],
            )
        })
        .unzip()
}

// -- separate file? --

fn constant<'c, 'a>(
    value: &uiua::Value,
    block: &'a Block<'c>,
    ctx: CompileContext<'c, '_>,
) -> Result<Value<'c, 'a>> {
    let loc = Location::unknown(ctx.context);

    let info = ValInfo::from_value(value.clone());
    let elem_type = mk_elem_type(&CompType::from_info(&info), ctx);

    let elem_attrs = if info.range.float
        && let Some(num_arr) = value.as_num_array()
    {
        num_arr
            .elements()
            .map(|&elem| FloatAttribute::new(ctx.context, elem_type, elem).into())
            .collect_vec()
    } else if let Some(ints) = value
        .as_num_array()
        .map(|arr| arr.elements().map(|&float| float as i64).collect_vec())
        .or_else(|| {
            value
                .as_byte_array()
                .map(|arr| arr.elements().map(|&byte| byte as i64).collect_vec())
        })
        .or_else(|| {
            value
                .as_char_array()
                .map(|arr| arr.elements().map(|&byte| byte as i64).collect_vec())
        })
    {
        ints.into_iter()
            .map(|elem| IntegerAttribute::new(elem_type, elem).into())
            .collect_vec()
    } else {
        unimplemented!()
    };

    let val_type = mk_tensor_type(&info.shape, elem_type);
    let dense_attr = DenseElementsAttribute::new(val_type, &elem_attrs)?;

    let op = arith::constant(ctx.context, val_type, dense_attr.into(), loc);
    Ok(block.append_operation(op.into()).result(0)?.into())
}

fn call<'c, 'a, 'u>(
    idx: NodeIndex,
    deps: &Stack,
    span: usize,
    block: &'a Block<'c>,
    fctx: &mut FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Vec<Value<'c, 'a>>> {
    let loc = span_to_loc(span, ctx);
    let comp_node = &fctx.compile_graph.node_weight(idx).unwrap().0;
    let analyzed_func = &fctx.funclib.funcs[comp_node.info.subfunc_idxs[0]];
    let func_name = name_mangle(analyzed_func)?;
    let ref_attr = FlatSymbolRefAttribute::new(ctx.context, &func_name);
    let args = deps
        .iter()
        .map(|&(idx, out_i)| {
            fctx.compile_graph
                .node_weight(idx)
                .and_then(|(_, v)| v.as_ref()?.get(out_i).copied())
        })
        .collect::<Option<Vec<_>>>()
        .expect("Argument missing from compile graph");

    let out_types = comp_node
        .info
        .vals
        .iter()
        .map(|out_info| mk_type(out_info, ctx))
        .collect_vec();

    let op = func::call(ctx.context, ref_attr, &args, &out_types, loc);
    let op_ref = block.append_operation(op);
    (0..out_types.len())
        .map(|i| op_ref.result(i).map(Into::into).map_err(Into::into))
        .collect::<Result<_>>()
}

fn upcast_num<'c, 'a, 'u>(
    cast: Cast,
    idx: NodeIndex,
    deps: &Stack,
    span: usize,
    block: &'a Block<'c>,
    fctx: &mut FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let loc = span_to_loc(span, ctx);
    let (_dep_infos, dep_vals) = get_deps(deps, fctx.compile_graph);
    let dep_val = dep_vals[0];

    let out_node = &fctx.compile_graph.node_weight(idx).unwrap().0;
    let out_comp_type = &out_node.types[0];
    let out_shape = &out_node.info.vals[0].shape;
    let out_type = mk_type_from_comp_shape(out_comp_type, out_shape, ctx);

    let op_name = match cast {
        Cast::UInt => "arith.extui",
        Cast::SInt => "arith.extsi",
        Cast::UtoF => "arith.uitofp",
        Cast::StoF => "arith.sitofp",
    };
    let op = OperationBuilder::new(op_name, loc)
        .add_results(&[out_type])
        .add_operands(&[dep_val])
        .build()?;

    Ok(block.append_operation(op).result(0)?.into())
}

fn print<'c, 'a, 'u>(
    deps: &Stack,
    span: usize,
    block: &'a Block<'c>,
    fctx: &mut FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<()> {
    let loc = span_to_loc(span, ctx);
    let (arg_idx, arg_out_i) = deps[0];
    let arg_info = &fctx.compile_graph.node_weight(arg_idx).unwrap().0.info.vals[arg_out_i];

    let comp_type = CompType::from_info(arg_info);
    let elem_type = mk_elem_type(&comp_type, ctx);
    let print_func = match comp_type {
        CompType::Int(_, i) => ["print_i8", "print_i16", "print_i32", "print_i64"][i as usize],
        CompType::Float(d) => {
            if d {
                "print_f64"
            } else {
                "print_f32"
            }
        }
        CompType::Bool => "print_i1",
        CompType::Char => "print_i32",
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
    let deshaped_type = RankedTensorType::new(&[len], elem_type, None);

    let arg_val = *fctx
        .compile_graph
        .node_weight(arg_idx)
        .unwrap()
        .1
        .as_ref()
        .and_then(|v| v.get(arg_out_i))
        .expect("Argument not compiled");

    let flat_shape_type = RankedTensorType::new(&[1], ctx.index_type, None).into();
    let flat_shape_val: Value = block
        .append_operation(
            arith::constant(
                ctx.context,
                flat_shape_type,
                DenseElementsAttribute::new(
                    flat_shape_type,
                    &[IntegerAttribute::new(ctx.index_type, len as i64).into()],
                )?
                .into(),
                loc,
            )
            .into(),
        )
        .result(0)?
        .into();

    let deshape_op = tensor::reshape(
        ctx.context,
        deshaped_type.into(),
        arg_val,
        flat_shape_val,
        loc,
    );

    let deshaped_val: Value = block.append_operation(deshape_op.into()).result(0)?.into();

    let zero_val: Value = block
        .append_operation(
            arith::constant(
                ctx.context,
                ctx.index_type,
                IntegerAttribute::new(ctx.index_type, 0).into(),
                loc,
            )
            .into(),
        )
        .result(0)?
        .into();
    let one_val: Value = block
        .append_operation(
            arith::constant(
                ctx.context,
                ctx.index_type,
                IntegerAttribute::new(ctx.index_type, 1).into(),
                loc,
            )
            .into(),
        )
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
            let get_op = tensor::extract(ctx.context, elem_type, deshaped_val, &[idx_val], loc);
            let cur_val: Value = for_block.append_operation(get_op.into()).result(0)?.into();
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

    Ok(())
}

fn add<'c, 'a, 'u>(
    idx: NodeIndex,
    deps: &Stack,
    span: usize,
    block: &'a Block<'c>,
    fctx: &mut FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let loc = span_to_loc(span, ctx);

    let (_dep_infos, dep_vals) = get_deps(deps, fctx.compile_graph);
    let (lhs_val, rhs_val) = (dep_vals[0], dep_vals[1]);

    let out_info = &fctx.compile_graph.node_weight(idx).unwrap().0.info.vals[0];
    let out_type = mk_type(out_info, ctx);

    let op = tosa::add(ctx.context, out_type, lhs_val, rhs_val, loc);

    Ok(block.append_operation(op.into()).result(0)?.into())
}
