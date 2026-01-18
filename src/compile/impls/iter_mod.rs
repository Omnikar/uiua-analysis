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
