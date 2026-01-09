use super::*;

pub fn perv_monad<'c, 'a, 'u>(
    op_name: &str,
    comp_node: &CompNode,
    deps: StackSlice,
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let loc = span_to_loc(span, ctx);

    let (_dep_infos, dep_vals) = get_deps(deps, fctx.compile_graph);
    let dep_val = dep_vals[0];

    let out_info = &comp_node.info.vals[0];
    let out_type = mk_type(out_info, ctx);

    let op_builder = OperationBuilder::new(op_name, loc)
        .add_results(&[out_type])
        .add_operands(&[dep_val]);

    let op = op_builder.build()?;

    Ok(block.append_operation(op).result(0)?.into())
}

pub fn sub_const<'c, 'a, 'u>(
    num: i64,
    comp_node: &CompNode,
    deps: StackSlice,
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let loc = span_to_loc(span, ctx);

    let (_dep_infos, dep_vals) = get_deps(deps, fctx.compile_graph);
    let dep_val = dep_vals[0];

    let out_info = &comp_node.info.vals[0];
    let out_type = mk_type(out_info, ctx);

    let dep_rtt = RankedTensorType::try_from(dep_val.r#type())?;
    let rank = dep_rtt.rank();
    let elem_type = dep_rtt.element();
    let num_tensor_type: Type = RankedTensorType::new(&vec![1; rank], elem_type, None).into();

    let num_val: Value = block
        .append_operation(
            arith::constant(
                ctx.context,
                num_tensor_type,
                DenseElementsAttribute::new(
                    num_tensor_type,
                    &[IntegerAttribute::new(elem_type, num).into()],
                )?
                .into(),
                loc,
            )
            .into(),
        )
        .result(0)?
        .into();

    let sub_op = tosa::sub(ctx.context, out_type, num_val, dep_val, loc);
    Ok(block.append_operation(sub_op.into()).result(0)?.into())
}
