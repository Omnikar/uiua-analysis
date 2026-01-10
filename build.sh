#!/bin/sh

set -e

cd build

dot -Tpng data-graph.dot -o data-graph.png
dot -Tpng compile-graph.dot -o compile-graph.png

# name=$1
name=test

LLVM_DIR=$(llvm-config --libdir)

mlir-opt $name.mlir --pass-pipeline='builtin.module(func.func(tosa-to-linalg),tosa-to-arith,convert-elementwise-to-linalg,convert-math-to-llvm,one-shot-bufferize{bufferize-function-boundaries},func.func(buffer-hoisting,buffer-loop-hoisting),drop-equivalent-buffer-results,func.func(promote-buffers-to-stack),buffer-deallocation-pipeline,convert-bufferization-to-memref,expand-strided-metadata,convert-linalg-to-affine-loops,lower-affine,convert-vector-to-llvm,finalize-memref-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,convert-index-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' -o $name_opt.mlir

mlir-translate $name_opt.mlir \
  -mlir-to-llvmir \
  -o $name_opt.ll

llc -filetype=obj --relocation-model=pic $name_opt.ll -o $name.o

clang $name.o \
  -L $LLVM_DIR \
  -lmlir_c_runner_utils \
  -o $name
