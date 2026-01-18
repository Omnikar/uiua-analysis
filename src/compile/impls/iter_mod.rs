use super::*;

pub fn sum_product<'c, 'a, 'u>(
    product: bool,
    comp_node: &CompNode,
    deps: &[(NodeIndex, usize)],
    span: usize,
    block: &'a Block<'c>,
    fctx: &mut FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let loc = span_to_loc(span, ctx);

    let (dep_infos, _dep_types, dep_vals) = get_deps(deps, fctx.compile_graph);
    let (dep_info, dep_val) = (dep_infos[0], dep_vals[0]);

    let out_type = mk_type_from_comp_shape(&comp_node.types[0], &comp_node.info.vals[0].shape, ctx);

    match dep_info.shape.rank() {
        Some(1..) => {}
        Some(0) => return Ok(dep_val),
        None => todo!("Reverse unranked tensor"),
    }

    let reduce_op: Operation = if product {
        tosa::reduce_product(
            ctx.context,
            dep_val,
            IntegerAttribute::new(ctx.int_types[2], 0),
            loc,
        )
        .into()
    } else {
        tosa::reduce_sum(
            ctx.context,
            dep_val,
            IntegerAttribute::new(ctx.int_types[2], 0),
            loc,
        )
        .into()
    };
    let sum_val = one_op_val(block, reduce_op)?;

    // - Extract one row -
    let shape = dep_info
        .shape
        .known_shape()
        .unwrap()
        .into_iter()
        .map(|x| x.map(|x| x as i64).unwrap_or(DYN_AX as i64))
        .collect_vec();

    let static_offsets = vec![0; shape.len()];
    let mut static_sizes = shape.clone();
    static_sizes[0] = 1;
    let static_strides = vec![1; shape.len()];

    let mut size_vals = Vec::<Value>::new();
    for (i, &dim) in shape.iter().enumerate().skip(1) {
        if dim as u64 == DYN_AX {
            let dim_i_val = const_int(i as i64, ctx.index_type, block, ctx, loc)?;
            let dim_op = tensor::dim(ctx.context, ctx.index_type, sum_val, dim_i_val, loc);
            let len_val = one_op_val(block, dim_op)?;
            size_vals.push(len_val);
        }
    }

    let attributes: Vec<(Identifier, Attribute)> = [
        ("static_offsets", &*static_offsets),
        ("static_sizes", &*static_sizes),
        ("static_strides", &*static_strides),
    ]
    .into_iter()
    .map(|(name, arr)| {
        (
            Identifier::new(ctx.context, name),
            DenseI64ArrayAttribute::new(ctx.context, arr).into(),
        )
    })
    .collect_vec();

    let get_op = OperationBuilder::new("tensor.extract_slice", loc)
        .add_results(&[out_type])
        .add_operands(&[sum_val])
        .add_operands(&size_vals)
        .add_attributes(&attributes)
        .add_attributes(&[(
            Identifier::new(ctx.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(ctx.context, &[1, 0, size_vals.len() as i32, 0]).into(),
        )])
        .build()?;
    // ---

    Ok(one_op_val(block, get_op)?)
}

pub fn do_loop<'c, 'a, 'u>(
    comp_node: &CompNode,
    deps: &[(NodeIndex, usize)],
    span: usize,
    block: &'a Block<'c>,
    fctx: &mut FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Vec<Value<'c, 'a>>> {
    let loc = span_to_loc(span, ctx);

    let (dep_infos, dep_comp_types, dep_vals) = get_deps(deps, fctx.compile_graph);

    let cond_idx = comp_node.info.subfunc_idxs[1];
    let (cond_graph, cond_info_map) = &fctx.func_infos.subfuncs[cond_idx];

    let body_idx = comp_node.info.subfunc_idxs[0];
    let (body_graph, body_info_map) = &fctx.func_infos.subfuncs[body_idx];

    let cond_in = cond_graph.arg_count();
    let cond_out = cond_graph.stack.len();
    let body_in = body_graph.arg_count();
    let body_out = body_graph.stack.len();

    let cond_block_sig = dep_comp_types
        .iter()
        .zip(&dep_infos)
        .map(|(comp_type, info)| mk_type_from_comp_shape(comp_type, &info.shape, ctx))
        .map(|typ| (typ, loc))
        .collect_vec();

    let cond_block = Block::new(&cond_block_sig);

    let cond_arg_vals: Vec<Value> = (0..dep_vals.len())
        .map(|i| cond_block.argument(i).map(Into::into).map_err(Into::into))
        .collect::<Result<_>>()?;
    let cond_pre_compile_graph =
        prepare_graph(cond_graph, cond_info_map, fctx.func_infos, ctx.uiua);
    let node_idxs = cond_pre_compile_graph.graph.node_indices().collect_vec();
    let mut cond_compile_graph = new_compile_graph(cond_pre_compile_graph.graph, &cond_arg_vals);

    let mut sub_fctx = FuncCompileContext {
        compile_graph: &mut cond_compile_graph,
        func_infos: fctx.func_infos,
        funclib: fctx.funclib,
    };

    for idx in node_idxs {
        compile_node(idx, &cond_block, &mut sub_fctx, ctx)?;
    }

    let mut to_pass = vals_from_cg(&cond_pre_compile_graph.stack, &cond_compile_graph)?;
    to_pass.extend_from_slice(&cond_arg_vals[cond_in..]);

    let mut cond_val = to_pass.remove(0);
    let elem_type = RankedTensorType::try_from(cond_val.r#type())?.element();
    let extract_op = tensor::extract(ctx.context, elem_type, cond_val, &[], loc);
    let scalar_val = one_op_val(&cond_block, extract_op)?;
    cond_val = scalar_val;
    let width = IntegerType::try_from(scalar_val.r#type())?.width();
    if width > 1 {
        let cast_op = arith::trunci(ctx.context, ctx.bool_type, scalar_val, loc);
        cond_val = one_op_val(&cond_block, cast_op)?;
    }

    let condition_op = scf::condition(ctx.context, cond_val, &to_pass, loc);
    cond_block.append_operation(condition_op.into());

    let body_block_sig = to_pass.iter().map(|val| (val.r#type(), loc)).collect_vec();

    let body_block = Block::new(&body_block_sig);

    let body_arg_vals: Vec<Value> = (0..to_pass.len())
        .map(|i| body_block.argument(i).map(Into::into).map_err(Into::into))
        .collect::<Result<_>>()?;

    let body_pre_compile_graph =
        prepare_graph(body_graph, body_info_map, fctx.func_infos, ctx.uiua);
    let node_idxs = body_pre_compile_graph.graph.node_indices().collect_vec();
    let mut body_compile_graph = new_compile_graph(body_pre_compile_graph.graph, &body_arg_vals);

    let mut sub_fctx = FuncCompileContext {
        compile_graph: &mut body_compile_graph,
        func_infos: fctx.func_infos,
        funclib: fctx.funclib,
    };

    for idx in node_idxs {
        compile_node(idx, &body_block, &mut sub_fctx, ctx)?;
    }

    let mut outs = vals_from_cg(&body_pre_compile_graph.stack, &body_compile_graph)?;
    outs.extend_from_slice(&body_arg_vals[body_in..]);

    let yield_op = scf::r#yield(ctx.context, &outs, loc);
    body_block.append_operation(yield_op.into());

    let cond_region = Region::new();
    cond_region.append_block(cond_block);
    let body_region = Region::new();
    body_region.append_block(body_block);

    let out_types = comp_node
        .types
        .iter()
        .zip(comp_node.info.vals.iter())
        .map(|(comp_type, info)| mk_type_from_comp_shape(comp_type, &info.shape, ctx))
        .collect_vec();

    let while_op = scf::r#while(
        ctx.context,
        &out_types,
        &dep_vals,
        cond_region,
        body_region,
        loc,
    );
    let op_ref = block.append_operation(while_op.into());

    (0..out_types.len())
        .map(|i| op_ref.result(i).map(Into::into).map_err(Into::into))
        .collect()
}
