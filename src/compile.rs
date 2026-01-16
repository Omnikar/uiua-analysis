mod impls;

use anyhow::{bail, Context as _, Result};
use itertools::Itertools;
use melior::{
    dialect::{
        func, index,
        ods::{arith, bufferization, llvm, memref, scf, tensor},
        DialectRegistry,
    },
    ir::{
        attribute::{
            DenseElementsAttribute, DenseI32ArrayAttribute, FlatSymbolRefAttribute,
            IntegerAttribute, StringAttribute, TypeAttribute,
        },
        operation::{OperationLike, OperationMutLike},
        r#type::{FunctionType, MemRefType, RankedTensorType},
        *,
    },
    pass,
    utility::register_all_dialects,
    Context,
};
use petgraph::{graph::NodeIndex, stable_graph::StableGraph};
use std::io::Write;
use uiua::{Node, Purity, SysOp};

use crate::{
    analyze::{
        analyze_func_graph, axis::Axis, AnalyzedFunc, FuncInfos, FuncLib, ShapeInfo, ValInfo,
    },
    graph::{Data, DataGraph, Stack, StackSlice},
    pre_compile::{prepare_graph, CompNode, CompType, Impl, Op},
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

    let module = Module::parse(ctx.context, include_str!("stdlib_header.mlir"))
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

    assert!(module.as_operation().verify());

    let mut f = std::fs::File::create("build/test.mlir")?;
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

    let mut pre_compile_graph = prepare_graph(&func.graph, &func.infos.map, ctx.uiua);

    // If this is the main function, then any leftover outputs get automatically connected to new pretty-print nodes in order to print them when the program ends
    if func_name == "main" {
        for (idx, out_i) in std::mem::replace(&mut pre_compile_graph.stack, Stack::new()) {
            let comp_node = CompNode {
                op: Op::Impl(Impl::EndShow, usize::MAX),
                info: crate::analyze::NodeInfo::no_vals(),
                types: smallvec::SmallVec::new(),
            };
            let new_idx = pre_compile_graph.graph.add_node(comp_node);
            pre_compile_graph.graph.add_edge(new_idx, idx, (out_i, 0));
        }
    }

    let mut compile_graph = new_compile_graph(pre_compile_graph.graph, &[]);

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
        vals_from_cg(&pre_compile_graph.stack, &compile_graph)?
    } else {
        vec![const_int(0, ctx.index_type, &block, ctx, loc)?]
    };
    block.append_operation(func::r#return(&outs, loc));

    let region = Region::new();
    region.append_block(block);

    // TODO: Figure out what attributes to put to indicate purity

    let mut func = func::func(
        ctx.context,
        StringAttribute::new(ctx.context, &func_name),
        TypeAttribute::new(FunctionType::new(ctx.context, &sig_in, &sig_out).into()),
        region,
        &[],
        loc,
    );

    Ok(func)
}

fn new_compile_graph<'c, 'a, 'u>(
    pre_compile_graph: StableGraph<CompNode<'u>, (usize, usize)>,
    arg_vals: &[Value<'c, 'a>],
) -> FuncCompileGraph<'c, 'a, 'u> {
    // pre_compile_graph.map_owned(|_, node| (node, None), |_, x| x)
    pre_compile_graph.map_owned(
        |_, node| {
            let val = match &node.op {
                Op::Data(Data::Arg(i)) => arg_vals.get(*i).map(|&x| vec![x]),
                _ => None,
            };
            (node, val)
        },
        |_, x| x,
    )
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

    let comp_node = fctx
        .compile_graph
        .node_weight(idx)
        .expect("Node missing from compile graph")
        .0
        .clone();

    use uiua::Primitive::*;
    let value = match comp_node.op {
        Op::Data(Data::Arg(i)) => vec![block.argument(i)?.into()],
        Op::Data(Data::Node(Node::Push(value))) => vec![impls::constant(value, block, ctx)?],
        Op::Data(Data::Node(Node::Call(_func, span))) => {
            impls::call(&comp_node, &deps, *span, block, fctx, ctx)?
        }

        Op::Impl(Impl::Cast(cast), span) => {
            vec![impls::cast_num(
                cast, &comp_node, &deps, span, block, fctx, ctx,
            )?]
        }

        // -- Monadic Pervasive Functions --
        Op::Data(Data::Node(Node::Prim(Not, span))) => {
            vec![impls::sub_const(
                1, &comp_node, &deps, *span, block, fctx, ctx,
            )?]
        }
        Op::Data(Data::Node(Node::Prim(Sign, span))) => todo!(),
        Op::Data(Data::Node(Node::Prim(Neg, span))) => {
            vec![impls::sub_const(
                0, &comp_node, &deps, *span, block, fctx, ctx,
            )?]
        }
        Op::Data(Data::Node(Node::Prim(Reciprocal, span))) => vec![impls::perv_monad(
            "tosa.reciprocal",
            &comp_node,
            &deps,
            *span,
            block,
            fctx,
            ctx,
        )?],
        Op::Data(Data::Node(Node::Prim(Abs, span))) => {
            vec![impls::perv_monad(
                "tosa.abs", &comp_node, &deps, *span, block, fctx, ctx,
            )?]
        }
        Op::Data(Data::Node(Node::Prim(Sqrt, span))) => {
            vec![impls::perv_monad(
                "math.sqrt",
                &comp_node,
                &deps,
                *span,
                block,
                fctx,
                ctx,
            )?]
        }
        Op::Data(Data::Node(Node::Prim(Exp, span))) => {
            vec![impls::perv_monad(
                "math.exp", &comp_node, &deps, *span, block, fctx, ctx,
            )?]
        }
        Op::Data(Data::Node(Node::Prim(Sin, span))) => {
            vec![impls::perv_monad(
                "math.sin", &comp_node, &deps, *span, block, fctx, ctx,
            )?]
        }
        Op::Data(Data::Node(Node::Prim(Floor, span))) => todo!(),
        Op::Data(Data::Node(Node::Prim(Ceil, span))) => todo!(),
        Op::Data(Data::Node(Node::Prim(Round, span))) => todo!(),

        // -- Dyadic Pervasive Functions --
        Op::Data(Data::Node(Node::Prim(Add, span))) => {
            vec![impls::arith(
                "tosa.add", &comp_node, &deps, *span, block, fctx, ctx,
            )?]
        }
        Op::Data(Data::Node(Node::Prim(Sub, span))) => {
            vec![impls::arith(
                "tosa.sub", &comp_node, &deps, *span, block, fctx, ctx,
            )?]
        }
        Op::Data(Data::Node(Node::Prim(Mul, span))) => {
            vec![impls::arith(
                "tosa.mul", &comp_node, &deps, *span, block, fctx, ctx,
            )?]
        }
        // FIXME: `arith` elementwise ops don't support fixing
        Op::Data(Data::Node(Node::Prim(Div, span))) => {
            vec![impls::arith(
                "arith.divf",
                &comp_node,
                &deps,
                *span,
                block,
                fctx,
                ctx,
            )?]
        }

        // -- Monadic Array Functions --
        Op::Data(Data::Node(Node::Prim(Range, span))) => {
            vec![impls::range(&comp_node, &deps, *span, block, fctx, ctx)?]
        }

        // -- Mapping Modifiers --
        Op::Data(Data::Node(Node::Mod(Rows, _funcs, span))) => {
            impls::rows(&comp_node, &deps, *span, block, fctx, ctx)?
        }

        Op::Data(Data::Node(Node::Prim(Sys(SysOp::Print), span))) => {
            // print(&deps, *span, block, fctx, ctx)?;
            show(&deps, *span, block, fctx, ctx)?;
            Vec::new()
        }
        Op::Data(Data::Node(&Node::Prim(Sys(SysOp::Show), span)))
        | Op::Impl(Impl::EndShow, span) => {
            show(&deps, span, block, fctx, ctx)?;
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
                CompType::from_info(info),
            )
        })
        .join("_");
    Ok(format!("_{base_name}__{name_suffix}"))
}

fn span_to_loc<'c>(span: usize, ctx: CompileContext<'c, '_>) -> Location<'c> {
    if let Some(sp) = ctx.uiua.asm.spans.get(span).cloned()
        && let Some(sp) = sp.code()
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

fn const_int<'a, 'c>(
    val: i64,
    typ: Type<'c>,
    block: &'a Block<'c>,
    ctx: CompileContext<'c, '_>,
    loc: Location<'c>,
) -> Result<Value<'c, 'a>> {
    block
        .append_operation(
            arith::constant(
                ctx.context,
                typ,
                IntegerAttribute::new(typ, val).into(),
                loc,
            )
            .into(),
        )
        .result(0)
        .map(Into::into)
        .map_err(Into::into)
}

fn vals_from_cg<'c, 'a, 'u, 'b>(
    idxs: impl IntoIterator<Item = &'b (NodeIndex, usize)>,
    cg: &FuncCompileGraph<'c, 'a, 'u>,
) -> Result<Vec<Value<'c, 'a>>> {
    idxs.into_iter()
        .map(|&(idx, out_i)| Some(cg.node_weight(idx)?.1.as_ref()?[out_i]))
        .collect::<Option<Vec<_>>>()
        .context("Did not compile required node")
}

fn tensor_to_unranked_memref<'c, 'a>(
    dep_info: &ValInfo,
    dep_val: Value<'c, 'a>,
    block: &'a Block<'c>,
    ctx: CompileContext<'c, '_>,
    loc: Location<'c>,
) -> Result<Value<'c, 'a>> {
    let dep_tensor_type = RankedTensorType::try_from(dep_val.r#type())?;
    let elem_type = dep_tensor_type.element();

    let dims = dims_from_shape_info(&dep_info.shape)
        .into_iter()
        .map(|x| x as i64)
        .collect_vec();

    let memref_type: Type = MemRefType::new(elem_type, &dims, None, None).into();

    let to_buffer_op = bufferization::to_buffer(ctx.context, memref_type, dep_val, loc);

    let null_attr = mlir_sys::MlirAttribute {
        ptr: std::ptr::null(),
    };
    let unranked_memref_type = unsafe {
        Type::from_raw(mlir_sys::mlirUnrankedMemRefTypeGet(
            elem_type.to_raw(),
            null_attr,
        ))
    };

    let memref_val: Value = block
        .append_operation(to_buffer_op.into())
        .result(0)?
        .into();

    let memref_cast_op = memref::cast(ctx.context, unranked_memref_type, memref_val, loc);

    let unranked_memref_val: Value = block
        .append_operation(memref_cast_op.into())
        .result(0)?
        .into();

    Ok(unranked_memref_val)
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

// -- separate file? --

fn print<'c, 'a, 'u>(
    deps: StackSlice,
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<()> {
    let loc = span_to_loc(span, ctx);
    let (arg_idx, arg_out_i) = deps[0];
    let arg_info = &fctx.compile_graph.node_weight(arg_idx).unwrap().0.info.vals[arg_out_i];

    let comp_type = CompType::from_info(arg_info);
    let elem_type = mk_elem_type(&comp_type, ctx);
    let print_func = match comp_type {
        CompType::Int(s, i) => [
            ["print_u8", "print_u16", "print_u32", "print_u64"],
            ["print_i8", "print_i16", "print_i32", "print_i64"],
        ][s as usize][i as usize],
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

    let zero_val = const_int(0, ctx.index_type, block, ctx, loc)?;
    let one_val = const_int(1, ctx.index_type, block, ctx, loc)?;

    let flat_shape_type = RankedTensorType::new(&[1], ctx.index_type, None).into();
    let flat_shape_val: Value = if len == DYN_AX {
        let rank = arg_info
            .shape
            .rank()
            .context("Cannot print unranked tensor")?;
        let rank_val = const_int(rank as i64, ctx.index_type, block, ctx, loc)?;

        let for_block = Block::new(&[(ctx.index_type, loc); 2]);
        let dim_i_val: Value = for_block.argument(0)?.into();
        let dim_op = tensor::dim(ctx.context, ctx.index_type, arg_val, dim_i_val, loc);
        let dim_val: Value = for_block.append_operation(dim_op.into()).result(0)?.into();
        let old_size_val: Value = for_block.argument(1)?.into();
        let mul_op = index::mul(old_size_val, dim_val, loc);
        let new_size_val: Value = for_block.append_operation(mul_op).result(0)?.into();
        for_block.append_operation(scf::r#yield(ctx.context, &[new_size_val], loc).into());
        let for_region = Region::new();
        for_region.append_block(for_block);
        let for_op = scf::r#for(
            ctx.context,
            &[ctx.index_type],
            zero_val,
            rank_val,
            one_val,
            &[one_val],
            for_region,
            loc,
        );

        let size_val: Value = block.append_operation(for_op.into()).result(0)?.into();

        block
            .append_operation(
                tensor::from_elements(ctx.context, flat_shape_type, &[size_val], loc).into(),
            )
            .result(0)?
            .into()
    } else {
        block
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
            .into()
    };

    let deshape_op = tensor::reshape(
        ctx.context,
        deshaped_type.into(),
        arg_val,
        flat_shape_val,
        loc,
    );

    let deshaped_val: Value = block.append_operation(deshape_op.into()).result(0)?.into();

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

            let mut print_op = llvm::call(
                ctx.context,
                &[cur_val],
                &[],
                DenseI32ArrayAttribute::new(ctx.context, &[]),
                loc,
            );
            print_op.set_callee(FlatSymbolRefAttribute::new(ctx.context, print_func));
            let mut print_op: Operation = print_op.into();
            print_op.set_attribute(
                "operandSegmentSizes",
                DenseI32ArrayAttribute::new(ctx.context, &[1, 0]).into(),
            );

            for_block.append_operation(print_op);
            for_block.append_operation(scf::r#yield(ctx.context, &[], loc).into());
            let region = Region::new();
            region.append_block(for_block);
            region
        },
        loc,
    );

    block.append_operation(for_op.into());

    let mut final_print_op = llvm::call(
        ctx.context,
        &[],
        &[],
        DenseI32ArrayAttribute::new(ctx.context, &[]),
        loc,
    );
    final_print_op.set_callee(FlatSymbolRefAttribute::new(ctx.context, "print_ln"));
    block.append_operation(final_print_op.into());

    Ok(())
}

fn show<'c, 'a, 'u>(
    deps: StackSlice,
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<()> {
    let loc = span_to_loc(span, ctx);

    let (dep_infos, _dep_types, dep_vals) = impls::get_deps(deps, fctx.compile_graph);
    let (dep_info, dep_val) = (dep_infos[0], dep_vals[0]);

    let unranked_memref_val = tensor_to_unranked_memref(dep_info, dep_val, block, ctx, loc)?;

    let comp_type = CompType::from_info(dep_info);
    let print_func_suffix = match comp_type {
        CompType::Int(s, i) => {
            [["u8", "u16", "u32", "u64"], ["i8", "i16", "i32", "i64"]][s as usize][i as usize]
        }
        CompType::Float(d) => {
            if d {
                "f64"
            } else {
                "f32"
            }
        }
        // TODO: Idk
        CompType::Bool => "i1",
        CompType::Char => "i32",
    };
    let print_func = format!("print_show_{print_func_suffix}");

    let call_op = func::call(
        ctx.context,
        FlatSymbolRefAttribute::new(ctx.context, &print_func),
        &[unranked_memref_val],
        &[],
        loc,
    );

    block.append_operation(call_op);

    Ok(())
}
