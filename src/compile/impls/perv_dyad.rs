use super::*;
use crate::{graph::StackSlice, pre_compile::CompNode};

pub fn arith<'c, 'a, 'u>(
    op_name: &str,
    comp_node: &CompNode,
    deps: StackSlice,
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Value<'c, 'a>> {
    let loc = span_to_loc(span, ctx);

    let (dep_infos, _dep_types, mut dep_vals) = get_deps(deps, fctx.compile_graph);

    match_ranks(&mut dep_vals, &dep_infos, loc, block, ctx)?;

    let (lhs_val, rhs_val) = (dep_vals[0], dep_vals[1]);

    let out_info = &comp_node.info.vals[0];
    let out_type = mk_type(out_info, ctx);

    let mut op_builder = OperationBuilder::new(op_name, loc)
        .add_results(&[out_type])
        .add_operands(&[rhs_val, lhs_val]);

    if op_name == "tosa.mul" {
        let scalar_type = RankedTensorType::new(&[1], ctx.int_types[0], None).into();
        let zero_val = one_op_val(
            block,
            arith::constant(
                ctx.context,
                scalar_type,
                DenseElementsAttribute::new(
                    scalar_type,
                    &[IntegerAttribute::new(ctx.int_types[0], 0).into()],
                )?
                .into(),
                loc,
            ),
        )?;
        op_builder = op_builder.add_operands(&[zero_val]);
    }

    let op = op_builder.build()?;

    Ok(one_op_val(block, op)?)
}
