mod arr_monad;
mod perv_dyad;
mod perv_monad;

mod map_mod;

pub use arr_monad::*;
pub use perv_dyad::*;
pub use perv_monad::*;

pub use map_mod::*;

use anyhow::{bail, Context as _, Result};
use itertools::Itertools;
use melior::{
    dialect::{
        cf, func, index,
        ods::{arith, tensor, tosa},
    },
    ir::{
        attribute::{
            DenseElementsAttribute, FlatSymbolRefAttribute, FloatAttribute, IntegerAttribute,
        },
        operation::OperationBuilder,
        r#type::RankedTensorType,
        *,
    },
};
use petgraph::graph::NodeIndex;

use super::{
    const_int, dims_from_shape_info, mk_elem_type, mk_tensor_type, mk_type,
    mk_type_from_comp_shape, name_mangle, span_to_loc, CompileContext, FuncCompileContext,
    FuncCompileGraph, DYN_AX,
};
use crate::{
    analyze::{ShapeInfo, ValInfo},
    graph::StackSlice,
    pre_compile::{Cast, CompNode, CompType},
};

pub fn constant<'c, 'a>(
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

pub fn call<'c, 'a, 'u>(
    comp_node: &CompNode,
    deps: StackSlice,
    span: usize,
    block: &'a Block<'c>,
    fctx: &FuncCompileContext<'c, 'a, 'u, '_, '_, '_>,
    ctx: CompileContext<'c, 'u>,
) -> Result<Vec<Value<'c, 'a>>> {
    let loc = span_to_loc(span, ctx);

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

pub fn cast_num<'c, 'a, 'u>(
    cast: Cast,
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

    let out_comp_type = &comp_node.types[0];
    let out_shape = &comp_node.info.vals[0].shape;
    let out_type = mk_type_from_comp_shape(out_comp_type, out_shape, ctx);

    let op = OperationBuilder::new(cast.into(), loc)
        .add_results(&[out_type])
        .add_operands(&[dep_val])
        .build()?;

    Ok(block.append_operation(op).result(0)?.into())
}

/// Returns `ValInfo`s and `Value`s for the dependencies at the given indices
fn get_deps<'c, 'a, 'cg>(
    deps: StackSlice,
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

fn match_ranks<'c, 'a>(
    vals: &mut [Value<'c, 'a>],
    infos: &[&ValInfo],
    loc: Location<'c>,
    block: &'a Block<'c>,
    ctx: CompileContext<'c, '_>,
) -> Result<()> {
    let max_rank = infos
        .iter()
        .map(|&info| info.shape.rank())
        .max()
        .context("Cannot match rank of no arrays")?
        .context("Cannot match rank of unranked array")?;

    for (val, &info) in vals.iter_mut().zip(infos) {
        enforce_min_rank(max_rank, val, &info.shape, loc, block, ctx)?;
    }

    Ok(())
}

fn enforce_min_rank<'c, 'a>(
    rank: usize,
    val: &mut Value<'c, 'a>,
    info: &ShapeInfo,
    loc: Location<'c>,
    block: &'a Block<'c>,
    ctx: CompileContext<'c, '_>,
) -> Result<()> {
    let mut shape = dims_from_shape_info(info);

    if shape.len() >= rank {
        return Ok(());
    }

    let rank_diff = rank - shape.len();

    shape.extend(std::iter::repeat_n(1, rank_diff));

    let elem_type = RankedTensorType::try_from(val.r#type())
        .context("Expected tensor-typed input to rank-match")?
        .element();

    let out_type = RankedTensorType::new(&shape, elem_type, None).into();

    let shape_type = RankedTensorType::new(&[rank as u64], ctx.index_type, None).into();
    let shape_val: Value = block
        .append_operation(tensor::empty(ctx.context, shape_type, &[], loc).into())
        .result(0)?
        .into();
    for (dim_i, &dim) in shape.iter().enumerate() {
        let dim_i_val = const_int(dim_i as i64, ctx.index_type, block, ctx, loc)?;
        let dim_val: Value = if dim == DYN_AX {
            let get_dim_op = tensor::dim(ctx.context, ctx.index_type, *val, dim_i_val, loc);
            block.append_operation(get_dim_op.into()).result(0)?.into()
        } else {
            const_int(dim as i64, ctx.index_type, block, ctx, loc)?
        };
        let insert_op = tensor::insert(
            ctx.context,
            shape_type,
            dim_val,
            shape_val,
            &[dim_i_val],
            loc,
        );
        block.append_operation(insert_op.into());
    }

    let reshape_op = tensor::reshape(ctx.context, out_type, *val, shape_val, loc);

    *val = block.append_operation(reshape_op.into()).result(0)?.into();

    Ok(())
}
