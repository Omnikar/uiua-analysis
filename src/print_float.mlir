module {
  llvm.func @printf(!llvm.ptr<0>, ...) -> i32
  llvm.mlir.global internal constant @_fmt("%f\n\00") : !llvm.array<4 x i8>

  func.func @print_f64(%0: f64) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<4 x i8>

    llvm.call @printf(%ptr, %0)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, f64) -> i32

    llvm.return
  }
}

