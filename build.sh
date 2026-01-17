#!/bin/sh

set -e

cd build

dot -Tpng data-graph.dot -o data-graph.png
dot -Tpng compile-graph.dot -o compile-graph.png

# name=$1
name=test

LLVM_DIR=$(llvm-config --libdir)

mlir-opt "$name".mlir --pass-pipeline='builtin.module(symbol-dce,cse,func.func(tosa-to-linalg),tosa-to-arith,convert-elementwise-to-linalg,one-shot-bufferize{bufferize-function-boundaries},func.func(buffer-hoisting,buffer-loop-hoisting),drop-equivalent-buffer-results,func.func(promote-buffers-to-stack),buffer-deallocation-pipeline,convert-bufferization-to-memref,expand-strided-metadata,convert-linalg-to-affine-loops,lower-affine,finalize-memref-to-llvm,convert-scf-to-cf,convert-to-llvm,reconcile-unrealized-casts,canonicalize)' -o "$name"_opt.mlir

mlir-translate "$name"_opt.mlir \
  -mlir-to-llvmir \
  -o "$name"_opt.ll

llc -filetype=obj --relocation-model=pic "$name"_opt.ll -o "$name".o

clang "$name".o \
  -L $LLVM_DIR \
  -lmlir_c_runner_utils \
  -L ../target/release \
  -L ../target/debug \
  -lstdlib \
  -o "$name"
