use super::*;

pub fn len<'c, 'a, 'u>(
    comp_node: &CompNode,
    deps: &[(NodeIndex, usize)],
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let loc = span_to_loc(span, ctx);

    let (dep_infos, _dep_types, dep_vals) = get_deps(deps, fctx.compile_graph);
    let (dep_info, dep_val) = (dep_infos[0], dep_vals[0]);

    let out_elem_type = mk_elem_type(&comp_node.types[0], ctx);
    let out_type: Type = RankedTensorType::new(&[], out_elem_type, None).into();

    match (
        dep_info
            .shape
            .len()
            .map(|opt| opt.map(|ax| ax.only_const())),
        1,
    ) {
        // Known first axis length or known scalar
        (Some(Some(Some(len))), _) | (Some(None), len) => {
            let scalar_val = const_int(len as i64, out_elem_type, block, ctx, loc)?;
            let tensor_op = tensor::from_elements(ctx.context, out_type, &[scalar_val], loc);
            let tensor_val = one_op_val(block, tensor_op)?;
            Ok(tensor_val)
        }
        // Known rank ≥1 with unknown first axis length
        (Some(Some(None)), _) => {
            let zero_val = const_int(0, ctx.index_type, block, ctx, loc)?;
            let dim_op = tensor::dim(ctx.context, ctx.index_type, dep_val, zero_val, loc);
            let dim_val = one_op_val(block, dim_op)?;
            let cast_op = index::castu(dim_val, out_elem_type, loc);
            let cast_val = one_op_val(block, cast_op)?;
            let tensor_op = tensor::from_elements(ctx.context, out_type, &[cast_val], loc);
            let tensor_val = one_op_val(block, tensor_op)?;
            Ok(tensor_val)
        }
        // Rank unknown
        (None, _) => todo!("Length of unranked tensor"),
    }
}

pub fn range<'c, 'a, 'u>(
    comp_node: &CompNode,
    deps: &[(NodeIndex, usize)],
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let (dep_infos, _dep_types, dep_vals) = get_deps(deps, fctx.compile_graph);
    let (dep_info, dep_val) = (dep_infos[0], dep_vals[0]);

    let rank = dep_info
        .shape
        .rank()
        .context("Cannot take range of unranked value")?;

    if rank == 0 {
        scalar_range(comp_node, dep_info, dep_val, span, block, ctx)
    } else {
        multidim_range(comp_node, dep_info, dep_val, span, block, ctx)
    }
}

fn scalar_range<'c, 'a, 'u>(
    comp_node: &CompNode,
    dep_info: &ValInfo,
    dep_val: Value<'c, 'a>,
    span: usize,
    block: &'a Block<'c>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let loc = span_to_loc(span, ctx);

    if dep_info
        .shape
        .rank()
        .context("Cannot take range of unranked value")?
        > 0
    {
        todo!()
    }

    let dim = match &dep_info.shape {
        ShapeInfo::Known(val) => val
            .as_num_array()
            .and_then(|arr| arr.as_scalar().map(|&n| n.abs() as u64))
            .or_else(|| {
                val.as_byte_array()
                    .and_then(|arr| arr.as_scalar().map(|&n| n as u64))
            })
            .unwrap(),
        _ => DYN_AX,
    };

    let out_comp_type = &comp_node.types[0];
    let out_elem_type = mk_elem_type(out_comp_type, ctx);
    let out_type = RankedTensorType::new(&[dim], out_elem_type, None).into();

    let signed = dep_info.range.signed;

    let mut dyn_len = Vec::new();
    let mut neg = None;

    if dim == DYN_AX || signed {
        let dep_elem_type = RankedTensorType::try_from(dep_val.r#type())?.element();
        let extract_op = tensor::extract(ctx.context, dep_elem_type, dep_val, &[], loc);
        let scalar_val = one_op_val(block, extract_op)?;

        if dim == DYN_AX {
            let abs_val: Value = if signed {
                one_op_val(
                    block,
                    tosa::abs(ctx.context, out_elem_type, scalar_val, loc),
                )?
            } else {
                scalar_val
            };
            let index_val = one_op_val(block, index::castu(abs_val, ctx.index_type, loc))?;
            dyn_len.push(index_val);
        }

        if signed {
            let zero_val = const_int(0, dep_elem_type, block, ctx, loc)?;

            let cmp_op = arith::cmpi(
                ctx.context,
                ctx.bool_type,
                scalar_val,
                zero_val,
                IntegerAttribute::new(ctx.int_types[3], 2).into(),
                loc,
            );
            let cmp_val = one_op_val(block, cmp_op)?;
            let cmp_casted_val = one_op_val(
                block,
                arith::extui(ctx.context, dep_elem_type, cmp_val, loc),
            )?;
            let neg_op = arith::subi(ctx.context, zero_val, cmp_casted_val, loc);
            let neg_val = one_op_val(block, neg_op)?;
            neg = Some(neg_val);
        }
    }

    let generate_block = Block::new(&[(ctx.index_type, loc)]);
    let coord_val: Value = generate_block.argument(0)?.into();
    let mut int_coord_val: Value = one_op_val(
        &generate_block,
        if signed {
            index::casts(coord_val, out_elem_type, loc)
        } else {
            index::castu(coord_val, out_elem_type, loc)
        },
    )?;
    if let Some(neg) = neg {
        int_coord_val = one_op_val(
            &generate_block,
            arith::xori(ctx.context, int_coord_val, neg, loc),
        )?;
    }
    generate_block.append_operation(tensor::r#yield(ctx.context, int_coord_val, loc).into());
    let generate_region = Region::new();
    generate_region.append_block(generate_block);

    let generate_op = tensor::generate(ctx.context, out_type, &dyn_len, generate_region, loc);

    let range_val = one_op_val(block, generate_op)?;

    Ok(range_val)
}

fn multidim_range<'c, 'a, 'u>(
    comp_node: &CompNode,
    dep_info: &ValInfo,
    dep_val: Value<'c, 'a>,
    span: usize,
    block: &'a Block<'c>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let loc = span_to_loc(span, ctx);

    let Some(coord_len) = dep_info
        .shape
        .len()
        .flatten()
        .and_then(|o| o.only_const())
        .map(|l| l as usize)
    else {
        unimplemented!();
    };

    let mut dims = match &dep_info.shape {
        ShapeInfo::Known(val) => val
            .as_num_array()
            .map(|arr| arr.elements().map(|&v| v.abs() as u64).collect_vec())
            .or_else(|| {
                val.as_byte_array()
                    .map(|arr| arr.elements().map(|&v| v as u64).collect_vec())
            })
            // })
            .unwrap(),
        _ => vec![DYN_AX; coord_len],
    };
    dims.push(coord_len as u64);

    let out_comp_type = &comp_node.types[0];
    let out_elem_type = mk_elem_type(out_comp_type, ctx);
    let out_type: Type = RankedTensorType::new(&dims, out_elem_type, None).into();

    let mut dyn_lens = Vec::new();
    let mut neg_mask = Vec::new();

    let dep_elem_type = RankedTensorType::try_from(dep_val.r#type())?.element();
    let signed = dep_info.range.signed;
    for (dim_i, &dim) in dims.iter().enumerate() {
        if dim != DYN_AX && !signed {
            continue;
        }
        let dim_i_val = const_int(dim_i as i64, ctx.index_type, block, ctx, loc)?;

        let extract_op = tensor::extract(ctx.context, dep_elem_type, dep_val, &[dim_i_val], loc);

        let scalar_val = one_op_val(block, extract_op)?;

        if dim == DYN_AX {
            let abs_val: Value = if signed {
                one_op_val(
                    block,
                    tosa::abs(ctx.context, out_elem_type, scalar_val, loc),
                )?
            } else {
                scalar_val
            };
            let index_val: Value = one_op_val(block, index::castu(abs_val, ctx.index_type, loc))?;
            dyn_lens.push(index_val);
        }

        if signed && dim_i < coord_len {
            let zero_val = const_int(0, dep_elem_type, block, ctx, loc)?;

            let cmp_op = arith::cmpi(
                ctx.context,
                ctx.bool_type,
                scalar_val,
                zero_val,
                IntegerAttribute::new(ctx.int_types[3], 2).into(),
                loc,
            );
            let cmp_val = one_op_val(block, cmp_op)?;
            let cmp_casted_val = one_op_val(
                block,
                arith::extui(ctx.context, dep_elem_type, cmp_val, loc),
            )?;
            let neg_op = arith::subi(ctx.context, zero_val, cmp_casted_val, loc);
            let neg_val = one_op_val(block, neg_op)?;
            neg_mask.push(neg_val);
        }
    }

    let block_args = vec![(ctx.index_type, loc); coord_len + 1];
    let generate_block = Block::new(&block_args);
    let mut coord_vals: Vec<Value> = (0..=coord_len)
        .map(|coord_i| generate_block.argument(coord_i).map(Into::into))
        .collect::<Result<_, _>>()?;
    let coord_idx_val = coord_vals.pop().unwrap();
    let coords_tensor_type: Type =
        RankedTensorType::new(&[coord_len as u64], ctx.index_type, None).into();
    let coords_tensor_val: Value = one_op_val(
        &generate_block,
        tensor::from_elements(ctx.context, coords_tensor_type, &coord_vals, loc),
    )?;

    let extract_op = tensor::extract(
        ctx.context,
        ctx.index_type,
        coords_tensor_val,
        &[coord_idx_val],
        loc,
    );
    let coord_val = one_op_val(&generate_block, extract_op)?;
    let int_coord_val: Value = one_op_val(
        &generate_block,
        if signed {
            index::casts(coord_val, out_elem_type, loc)
        } else {
            index::castu(coord_val, out_elem_type, loc)
        },
    )?;
    generate_block.append_operation(tensor::r#yield(ctx.context, int_coord_val, loc).into());

    let generate_region = Region::new();
    generate_region.append_block(generate_block);
    let generate_op = tensor::generate(ctx.context, out_type, &dyn_lens, generate_region, loc);

    let mut range_val = one_op_val(block, generate_op)?;

    if signed {
        let mut neg_mask_dims = vec![1; coord_len];
        neg_mask_dims.push(neg_mask.len() as u64);
        let neg_mask_type: Type = RankedTensorType::new(&neg_mask_dims, out_elem_type, None).into();
        let neg_mask_val: Value = one_op_val(
            block,
            tensor::from_elements(ctx.context, neg_mask_type, &neg_mask, loc),
        )?;
        let xor_op = tosa::bitwise_xor(ctx.context, out_type, range_val, neg_mask_val, loc);
        range_val = one_op_val(block, xor_op)?;
    }

    Ok(range_val)
}

pub fn first<'c, 'a, 'u>(
    comp_node: &CompNode,
    deps: &[(NodeIndex, usize)],
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let loc = span_to_loc(span, ctx);

    let (dep_infos, _dep_types, dep_vals) = get_deps(deps, fctx.compile_graph);
    let (dep_info, dep_val) = (dep_infos[0], dep_vals[0]);

    let out_type = mk_type_from_comp_shape(&comp_node.types[0], &comp_node.info.vals[0].shape, ctx);

    match dep_info
        .shape
        .len()
        .map(|opt| opt.map(|ax| ax.only_const()))
    {
        // Known rank ≥1 with known first axis length
        Some(Some(Some(len))) => {
            if len == 0 {
                bail!("Cannot take first of an empty array");
            }
        }
        // Known rank ≥1 with unknown first axis length
        Some(Some(None)) => {
            // Check that the array has at least one row
            let zero_val = const_int(0, ctx.index_type, block, ctx, loc)?;
            let dim_op = tensor::dim(ctx.context, ctx.index_type, dep_val, zero_val, loc);
            let dim_val = one_op_val(block, dim_op)?;
            let cmp_op = index::cmp(ctx.context, CmpiPredicate::Ugt, dim_val, zero_val, loc);
            let cmp_val = one_op_val(block, cmp_op)?;
            let err_msg = format!("{loc}: Cannot take first of an empty array");
            let assert_op = cf::assert(ctx.context, cmp_val, &err_msg, loc);
            block.append_operation(assert_op);
        }
        // Known scalar
        Some(None) => return Ok(dep_val),
        // Rank unknown
        None => todo!("First of unranked tensor"),
    }

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
            let dim_op = tensor::dim(ctx.context, ctx.index_type, dep_val, dim_i_val, loc);
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
        .add_operands(&[dep_val])
        .add_operands(&size_vals)
        .add_attributes(&attributes)
        .add_attributes(&[(
            Identifier::new(ctx.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(ctx.context, &[1, 0, size_vals.len() as i32, 0]).into(),
        )])
        .build()?;

    Ok(one_op_val(block, get_op)?)
}

pub fn reverse<'c, 'a, 'u>(
    comp_node: &CompNode,
    deps: &[(NodeIndex, usize)],
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
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

    let reverse_op = tosa::reverse(
        ctx.context,
        out_type,
        dep_val,
        IntegerAttribute::new(ctx.int_types[2], 0),
        loc,
    );
    Ok(one_op_val(block, reverse_op)?)
}
