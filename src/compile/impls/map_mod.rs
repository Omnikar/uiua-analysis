use super::*;

pub fn rows<'c, 'a, 'u>(
    comp_node: &CompNode,
    deps: &[(NodeIndex, usize)],
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let loc = span_to_loc(span, ctx);

    let (dep_infos, dep_vals) = get_deps(deps, fctx.compile_graph);

    let (subfunc_graph, subfunc_info_map) =
        &fctx.func_infos.subfuncs[comp_node.info.subfunc_idxs[0]];

    let zero_val = const_int(0, ctx.index_type, block, ctx, loc)?;
    let one_val = const_int(1, ctx.index_type, block, ctx, loc)?;

    let mut deps_fixed = vec![false; deps.len()];

    let out_len: Option<(Option<usize>, Value)> = if deps.is_empty() {
        None
    } else if deps.len() == 1 {
        match dep_infos[0].shape.len() {
            Some(Some(ax)) => match ax.only_const() {
                Some(len) => {
                    let len = len as usize;
                    let len_val = const_int(len as i64, ctx.index_type, block, ctx, loc)?;
                    Some((Some(len), len_val))
                }
                None => {
                    let dim_op =
                        tensor::dim(ctx.context, ctx.index_type, dep_vals[0], zero_val, loc);
                    let len_val = block.append_operation(dim_op.into()).result(0)?.into();
                    Some((None, len_val))
                }
            },
            Some(None) => None,
            None => bail!("Rows is not currently supported for unranked arrays"),
        }
    } else {
        // None: unknown length
        // Some(None): known to be a scalar
        // Some(Some(…)): statically known length
        let lens = dep_infos
            .iter()
            .map(|info| match info.shape.len() {
                // None: unknown rank
                // Some(None): known to be of rank ≥1, with unknown length
                // Some(Some(None)): known at compile time to be a scalar
                // Some(Some(Some(…))): rank ≥1 with a length known at compile time
                Some(Some(ax)) => Some(ax.only_const().map(|len| len as usize).map(Some)),
                Some(None) => Some(Some(None)),
                None => None,
            })
            .collect::<Option<Vec<_>>>()
            .context("Rows is not currently supported for unranked arrays")?;

        // The length against which to compare all other axes
        let ref_len: (Option<usize>, Value);
        // The index of the argument with that length
        let ref_len_i: usize;

        // Reference a statically known length if possible
        // Skip statically known 1-lengths since they distribute
        if let Some((i, known_ref_len)) = lens
            .iter()
            .copied()
            .map(Option::flatten)
            .enumerate()
            .find_map(|(i, x)| x.filter(|x| *x != 1).map(|x| (i, x)))
        {
            let ref_len_val = const_int(known_ref_len as i64, ctx.index_type, block, ctx, loc)?;
            ref_len = (Some(known_ref_len), ref_len_val);
            ref_len_i = i;
        }
        // If there are no non-length-1 statically known lengths, use the first unknown length
        else if let Some(i) = lens.iter().position(Option::is_none) {
            let dim_op = tensor::dim(ctx.context, ctx.index_type, dep_vals[i], zero_val, loc);
            let len_val = block.append_operation(dim_op.into()).result(0)?.into();
            ref_len = (None, len_val);
            ref_len_i = i;
        }
        // If neither of the above branches succeed, it can only be the case that all lengths are statically known to be 1
        else {
            ref_len = (Some(1), one_val);
            ref_len_i = 0;
            deps_fixed[0] = true;
        }

        for dep_i in (0..deps.len()).filter(|&i| i != ref_len_i) {
            let len: (Option<usize>, Value) = match lens[dep_i] {
                Some(Some(len)) => {
                    let len_val = const_int(len as i64, ctx.index_type, block, ctx, loc)?;
                    (Some(len), len_val)
                }
                Some(None) => continue, // Scalars automatically pass
                None => {
                    let dim_op =
                        tensor::dim(ctx.context, ctx.index_type, dep_vals[dep_i], zero_val, loc);
                    let len_val: Value = block.append_operation(dim_op.into()).result(0)?.into();
                    (None, len_val)
                }
            };

            if let Some(len) = len.0
                && len == 1
            {
                deps_fixed[dep_i] = true;
            }

            if let Some(ref_len) = ref_len.0
                && let Some(len) = len.0
            {
                if ref_len == len || len == 1 {
                    continue;
                } else {
                    bail!("Lengths {ref_len} and {len} are not compatible");
                }
            }

            let cmp_op = arith::cmpi(
                ctx.context,
                ctx.bool_type,
                ref_len.1,
                len.1,
                IntegerAttribute::new(ctx.int_types[3], 0).into(),
                loc,
            );
            let cmp_val: Value = block.append_operation(cmp_op.into()).result(0)?.into();

            let assert_op = cf::assert(ctx.context, cmp_val, "Array lengths are incompatible", loc);
            block.append_operation(assert_op);
        }

        Some(ref_len)
    };

    // TODO
    // If out_len is None, just call the function directly
    // Otherwise do rows stuff

    todo!()

    // Ok(block
    //     .append_operation(
    //         arith::constant(
    //             ctx.context,
    //             RankedTensorType::new(&[1], ctx.bool_type, None).into(),
    //             DenseElementsAttribute::new(
    //                 RankedTensorType::new(&[1], ctx.bool_type, None).into(),
    //                 &[IntegerAttribute::new(ctx.bool_type, 0).into()],
    //             )?
    //             .into(),
    //             loc,
    //         )
    //         .into(),
    //     )
    //     .result(0)?
    //     .into())
}
