use super::*;

pub fn range<'c, 'a, 'u>(
    comp_node: &CompNode,
    deps: &[(NodeIndex, usize)],
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let (dep_infos, dep_vals) = get_deps(deps, fctx.compile_graph);
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
        let scalar_val: Value = block.append_operation(extract_op.into()).result(0)?.into();

        if dim == DYN_AX {
            let abs_val: Value = if signed {
                block
                    .append_operation(tosa::abs(ctx.context, out_elem_type, scalar_val, loc).into())
                    .result(0)?
                    .into()
            } else {
                scalar_val
            };
            let index_val: Value = block
                .append_operation(index::castu(abs_val, ctx.index_type, loc))
                .result(0)?
                .into();
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
            let cmp_val: Value = block.append_operation(cmp_op.into()).result(0)?.into();
            let cmp_casted_val: Value = block
                .append_operation(arith::extui(ctx.context, dep_elem_type, cmp_val, loc).into())
                .result(0)?
                .into();
            let neg_op = arith::subi(ctx.context, zero_val, cmp_casted_val, loc);
            let neg_val: Value = block.append_operation(neg_op.into()).result(0)?.into();
            neg = Some(neg_val);
        }
    }

    let generate_block = Block::new(&[(ctx.index_type, loc)]);
    let coord_val: Value = generate_block.argument(0)?.into();
    let mut int_coord_val: Value = generate_block
        .append_operation(if signed {
            index::casts(coord_val, out_elem_type, loc)
        } else {
            index::castu(coord_val, out_elem_type, loc)
        })
        .result(0)?
        .into();
    if let Some(neg) = neg {
        int_coord_val = generate_block
            .append_operation(arith::xori(ctx.context, int_coord_val, neg, loc).into())
            .result(0)?
            .into();
    }
    generate_block.append_operation(tensor::r#yield(ctx.context, int_coord_val, loc).into());
    let generate_region = Region::new();
    generate_region.append_block(generate_block);

    let generate_op = tensor::generate(ctx.context, out_type, &dyn_len, generate_region, loc);

    let range_val: Value = block.append_operation(generate_op.into()).result(0)?.into();

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

        let scalar_val: Value = block.append_operation(extract_op.into()).result(0)?.into();

        if dim == DYN_AX {
            let abs_val: Value = if signed {
                block
                    .append_operation(tosa::abs(ctx.context, out_elem_type, scalar_val, loc).into())
                    .result(0)?
                    .into()
            } else {
                scalar_val
            };
            let index_val: Value = block
                .append_operation(index::castu(abs_val, ctx.index_type, loc))
                .result(0)?
                .into();
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
            let cmp_val: Value = block.append_operation(cmp_op.into()).result(0)?.into();
            let cmp_casted_val: Value = block
                .append_operation(arith::extui(ctx.context, dep_elem_type, cmp_val, loc).into())
                .result(0)?
                .into();
            let neg_op = arith::subi(ctx.context, zero_val, cmp_casted_val, loc);
            let neg_val: Value = block.append_operation(neg_op.into()).result(0)?.into();
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
    let coords_tensor_val: Value = generate_block
        .append_operation(
            tensor::from_elements(ctx.context, coords_tensor_type, &coord_vals, loc).into(),
        )
        .result(0)?
        .into();

    let extract_op = tensor::extract(
        ctx.context,
        ctx.index_type,
        coords_tensor_val,
        &[coord_idx_val],
        loc,
    );
    let coord_val: Value = generate_block
        .append_operation(extract_op.into())
        .result(0)?
        .into();
    let int_coord_val: Value = generate_block
        .append_operation(if signed {
            index::casts(coord_val, out_elem_type, loc)
        } else {
            index::castu(coord_val, out_elem_type, loc)
        })
        .result(0)?
        .into();
    generate_block.append_operation(tensor::r#yield(ctx.context, int_coord_val, loc).into());

    let generate_region = Region::new();
    generate_region.append_block(generate_block);
    let generate_op = tensor::generate(ctx.context, out_type, &dyn_lens, generate_region, loc);

    let mut range_val: Value = block.append_operation(generate_op.into()).result(0)?.into();

    if signed {
        let mut neg_mask_dims = vec![1; coord_len];
        neg_mask_dims.push(neg_mask.len() as u64);
        let neg_mask_type: Type = RankedTensorType::new(&neg_mask_dims, out_elem_type, None).into();
        let neg_mask_val: Value = block
            .append_operation(
                tensor::from_elements(ctx.context, neg_mask_type, &neg_mask, loc).into(),
            )
            .result(0)?
            .into();
        let xor_op = tosa::bitwise_xor(ctx.context, out_type, range_val, neg_mask_val, loc);
        range_val = block.append_operation(xor_op.into()).result(0)?.into();
    }

    Ok(range_val)
}
