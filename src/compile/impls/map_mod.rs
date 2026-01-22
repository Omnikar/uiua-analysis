use super::*;

pub fn rows<'c, 'a, 'u>(
    comp_node: &CompNode,
    deps: &[(NodeIndex, usize)],
    span: usize,
    block: &'a Block<'c>,
    fctx: &mut FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Vec<Value<'c, 'a>>> {
    let unranked_msg = "Rows is not currently supported for unranked arrays";

    let loc = span_to_loc(span, ctx);

    let (dep_infos, dep_comp_types, mut dep_vals) = get_deps(deps, fctx.compile_graph);

    let zero_val = const_int(0, ctx.index_type, block, ctx, loc)?;
    let one_val = const_int(1, ctx.index_type, block, ctx, loc)?;

    // - Identify fixed arguments and determine the number of rows to operate on -

    let mut deps_fixed = vec![false; deps.len()];

    let out_len: Option<(Option<usize>, Value)> = if deps.is_empty() {
        // When there are no arguments, the function is to be simply called once
        None
    } else if deps.len() == 1 {
        // If there is one argument, it provides the row count
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
                    let len_val = one_op_val(block, dim_op)?;
                    Some((None, len_val))
                }
            },
            Some(None) => None,
            None => bail!(unranked_msg),
        }
    } else {
        // Multiple arguments; do whatever necessary to ensure length matching and determine the output length

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
            .context(unranked_msg)?;

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
            let len_val = one_op_val(block, dim_op)?;
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
                Some(None) => {
                    // Scalars automatically pass
                    deps_fixed[dep_i] = true;
                    continue;
                }
                None => {
                    let dim_op =
                        tensor::dim(ctx.context, ctx.index_type, dep_vals[dep_i], zero_val, loc);
                    let len_val = one_op_val(block, dim_op)?;
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
            let cmp_val = one_op_val(block, cmp_op)?;

            let err_msg = format!("{loc}: Array lengths are incompatible");
            let assert_op = cf::assert(ctx.context, cmp_val, &err_msg, loc);
            block.append_operation(assert_op);
        }

        Some(ref_len)
    };

    // TODO
    // If out_len is None, just call the function directly
    // Otherwise do rows stuff

    let (subfunc_graph, subfunc_info_map) =
        &fctx.func_infos.subfuncs[comp_node.info.subfunc_idxs[0]];
    let pre_compile_graph =
        prepare_graph(subfunc_graph, subfunc_info_map, fctx.func_infos, ctx.uiua);

    let Some(out_len) = out_len else {
        // If out_len is None, all inputs are scalars and the function should be called directly
        let mut compile_graph = new_compile_graph(pre_compile_graph.graph, &dep_vals);
        // If all scalars, just call the function
        let mut sub_fctx = FuncCompileContext {
            compile_graph: &mut compile_graph,
            ..*fctx
        };

        for idx in sub_fctx.compile_graph.node_indices().collect_vec() {
            compile_node(idx, block, &mut sub_fctx, ctx)?;
        }

        return vals_from_cg(&pre_compile_graph.stack, &compile_graph);
    };

    // Promote all scalars to singleton lists
    for (info, val) in dep_infos.iter().zip(&mut dep_vals) {
        enforce_min_rank(1, val, &info.shape, loc, block, ctx)?;
    }

    // If code execution reaches here, at least one input is rank ≥1
    // Therefore, all outputs should also be rank ≥1

    let (dep_row_dims, dep_row_types): (Vec<Vec<u64>>, Vec<Type>) = dep_comp_types
        .iter()
        .zip(&dep_infos)
        .map(|(&comp_type, &val_info)| {
            let mut dims = val_info
                .shape
                .known_shape()
                .unwrap()
                .into_iter()
                .map(|x| x.map(|x| x as u64).unwrap_or(DYN_AX))
                .collect_vec();
            if !dims.is_empty() {
                dims.remove(0);
            }
            let elem_type = mk_elem_type(comp_type, ctx);
            let tensor_type: Type = RankedTensorType::new(&dims, elem_type, None).into();
            (dims, tensor_type)
        })
        .unzip();

    let out_row_shapes: Vec<Vec<u64>> = comp_node
        .info
        .vals
        .iter()
        .map(|val_info| {
            val_info.shape.known_shape().map(|sh| {
                sh.into_iter()
                    .skip(1)
                    .map(|len| len.map(|len| len as u64).unwrap_or(DYN_AX))
                    .collect_vec()
            })
        })
        .collect::<Option<_>>()
        .context(unranked_msg)?;

    let all_row_shapes_known: bool = out_row_shapes
        .iter()
        .flat_map(|v| v.iter())
        .all(|&x| x != DYN_AX);

    // If it is known at compile time that there are zero rows
    if out_len.0 == Some(0) {
        todo!()
    }

    if !all_row_shapes_known && out_len.0.is_none() {
        todo!()
    }

    let mut out_row_types = Vec::<Type>::new();
    let mut out_types = Vec::<Type>::new();
    let mut out_inits = Vec::<Value>::new();
    let start_i: usize;

    if all_row_shapes_known {
        for (row_shape, comp_type) in out_row_shapes.iter().zip(&comp_node.types) {
            let elem_type = mk_elem_type(comp_type, ctx);
            let row_type: Type = RankedTensorType::new(row_shape, elem_type, None).into();

            let mut out_shape = row_shape.clone();
            out_shape.insert(0, out_len.0.map(|x| x as u64).unwrap_or(DYN_AX));
            let out_type: Type = RankedTensorType::new(&out_shape, elem_type, None).into();

            let mut dyn_dims = Vec::new();
            if out_len.0.is_none() {
                dyn_dims.push(out_len.1);
            }

            let empty_op = tensor::empty(ctx.context, out_type, &dyn_dims, loc);
            let empty_val = one_op_val(block, empty_op)?;

            out_row_types.push(row_type);
            out_types.push(out_type);
            out_inits.push(empty_val);
        }

        start_i = 0;
    } else {
        todo!()
    }

    // - For loop -

    let for_block_args = std::iter::once(ctx.index_type)
        .chain(out_types.iter().copied())
        .map(|typ| (typ, loc))
        .collect_vec();
    let for_block = Block::new(&for_block_args);
    let idx_val: Value = for_block.argument(0)?.into();
    let accs: Vec<Value> = (1..=out_inits.len())
        .map(|i| Ok(for_block.argument(i).map(Into::into)?))
        .collect::<Result<_>>()?;

    let extracted: Vec<Value> = (0..deps.len())
        .map(|arg_i| {
            let dep_val = dep_vals[arg_i];
            // let dep_info = dep_infos[arg_i];

            let row_dims = &dep_row_dims[arg_i];
            let row_type = dep_row_types[arg_i];

            let mut static_offsets = vec![0; row_dims.len()];
            static_offsets.insert(0, DYN_AX as i64);

            let mut size_vals = Vec::<Value>::new();
            let mut static_sizes = vec![1];
            for (i, &dim) in row_dims.iter().enumerate() {
                static_sizes.push(dim as i64);
                if dim == DYN_AX {
                    let dim_i_val: Value =
                        const_int(i as i64 + 1, ctx.index_type, &for_block, ctx, loc)?;

                    let dim_op = tensor::dim(ctx.context, ctx.index_type, dep_val, dim_i_val, loc);
                    let len_val = one_op_val(&for_block, dim_op)?;

                    size_vals.push(len_val);
                }
            }

            let static_strides = vec![1; row_dims.len() + 1];

            let attributes: Vec<(Identifier, Attribute)> = [
                ("static_offsets", &*static_offsets),
                ("static_sizes", &*static_sizes),
                ("static_strides", &*static_strides),
                // ("operandSegmentSizes", &[1, 1, size_vals.len() as i64, 0]),
            ]
            .into_iter()
            .map(|(name, arr)| {
                (
                    Identifier::new(ctx.context, name),
                    DenseI64ArrayAttribute::new(ctx.context, arr).into(),
                )
            })
            .collect_vec();

            // If this argument is fixed, index at 0 instead of the current row index
            let idx_val = if deps_fixed[arg_i] { zero_val } else { idx_val };

            let get_op = OperationBuilder::new("tensor.extract_slice", loc)
                .add_results(&[row_type])
                .add_operands(&[dep_val, idx_val])
                .add_operands(&size_vals)
                .add_attributes(&attributes)
                .add_attributes(&[(
                    Identifier::new(ctx.context, "operandSegmentSizes"),
                    DenseI32ArrayAttribute::new(ctx.context, &[1, 1, size_vals.len() as i32, 0])
                        .into(),
                )])
                .build()?;

            Ok(one_op_val(&for_block, get_op)?)
        })
        .collect::<Result<_>>()?;

    let mut compile_graph = new_compile_graph(pre_compile_graph.graph, &extracted);
    let mut sub_fctx = FuncCompileContext {
        compile_graph: &mut compile_graph,
        ..*fctx
    };

    for idx in sub_fctx.compile_graph.node_indices().collect_vec() {
        compile_node(idx, &for_block, &mut sub_fctx, ctx)?;
    }

    let insert_vals = vals_from_cg(&pre_compile_graph.stack, &compile_graph)?;

    let mut yield_vals = Vec::<Value>::new();
    for (insert_val, acc, row_dims) in itertools::multizip((insert_vals, accs, &out_row_shapes)) {
        let mut static_offsets = vec![0; row_dims.len()];
        static_offsets.insert(0, DYN_AX as i64);

        let mut size_vals = Vec::<Value>::new();
        let mut static_sizes = vec![1];
        for (i, &dim) in row_dims.iter().enumerate() {
            static_sizes.push(dim as i64);
            if dim == DYN_AX {
                let dim_i_val: Value =
                    const_int(i as i64 + 1, ctx.index_type, &for_block, ctx, loc)?;

                let dim_op = tensor::dim(ctx.context, ctx.index_type, insert_val, dim_i_val, loc);
                let len_val = one_op_val(&for_block, dim_op)?;

                size_vals.push(len_val);
            }
        }

        let static_strides = vec![1; row_dims.len() + 1];

        let attributes: Vec<(Identifier, Attribute)> = [
            ("static_offsets", &*static_offsets),
            ("static_sizes", &*static_sizes),
            ("static_strides", &*static_strides),
            // ("operandSegmentSizes", &[1, 1, 1, size_vals.len() as i64, 0]),
        ]
        .into_iter()
        .map(|(name, arr)| {
            (
                Identifier::new(ctx.context, name),
                DenseI64ArrayAttribute::new(ctx.context, arr).into(),
            )
        })
        .collect_vec();

        let insert_op = OperationBuilder::new("tensor.insert_slice", loc)
            .add_results(&[acc.r#type()])
            .add_operands(&[insert_val, acc, idx_val])
            .add_operands(&size_vals)
            .add_attributes(&attributes)
            .add_attributes(&[(
                Identifier::new(ctx.context, "operandSegmentSizes"),
                DenseI32ArrayAttribute::new(ctx.context, &[1, 1, 1, size_vals.len() as i32, 0])
                    .into(),
            )])
            .build()?;

        let out_acc = one_op_val(&for_block, insert_op)?;
        yield_vals.push(out_acc);
    }

    for_block.append_operation(scf::r#yield(ctx.context, &yield_vals, loc).into());

    let for_region = Region::new();
    for_region.append_block(for_block);

    let for_op = OperationBuilder::new("scf.for", loc)
        .add_results(&out_types)
        .add_operands(&[
            const_int(start_i as i64, ctx.index_type, block, ctx, loc)?,
            out_len.1,
            one_val,
        ])
        .add_operands(&out_inits)
        .add_attributes(&[(
            Identifier::new(ctx.context, "operandSegmentSizes"),
            DenseI32ArrayAttribute::new(ctx.context, &[1, 1, 1, out_inits.len() as i32]).into(),
        )])
        .add_regions([for_region])
        .build()?;

    let op_ref = block.append_operation(for_op);

    (0..out_inits.len())
        .map(|i| op_ref.result(i).map(Into::into))
        .collect::<Result<_, _>>()
        .map_err(Into::into)
}
