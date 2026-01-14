module {
  // func.func private @example_func(memref<*xi8>) -> ()

  llvm.func @printf(!llvm.ptr<0>, ...) -> i32
  llvm.mlir.global internal constant @_ln_fmt("\n\00") : !llvm.array<2 x i8>
  llvm.mlir.global internal constant @_int_fmt("  %d\00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @_uint_fmt("  %u\00") : !llvm.array<5 x i8>
  llvm.mlir.global internal constant @_long_fmt("  %ld\00") : !llvm.array<6 x i8>
  llvm.mlir.global internal constant @_ulong_fmt("  %lu\00") : !llvm.array<6 x i8>
  llvm.mlir.global internal constant @_float_fmt("  %f\00") : !llvm.array<5 x i8>

  llvm.func @print_ln() -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_ln_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<2 x i8>

    llvm.call @printf(%ptr)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>) -> i32

    llvm.return
  }

  llvm.func @print_f32(%0: f32) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_float_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<5 x i8>

    llvm.call @printf(%ptr, %0)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, f32) -> i32

    llvm.return
  }

  llvm.func @print_f64(%0: f64) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_float_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<5 x i8>

    llvm.call @printf(%ptr, %0)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, f64) -> i32

    llvm.return
  }

  llvm.func @print_i1(%0: i1) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_int_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<5 x i8>

    llvm.call @printf(%ptr, %0)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, i1) -> i32

    llvm.return
  }

  llvm.func @print_u8(%0: i8) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_uint_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<5 x i8>

    %1 = llvm.zext %0 : i8 to i32
    llvm.call @printf(%ptr, %1)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, i32) -> i32

    llvm.return
  }

  llvm.func @print_i8(%0: i8) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_int_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<5 x i8>

    %1 = llvm.sext %0 : i8 to i32
    llvm.call @printf(%ptr, %1)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, i32) -> i32

    llvm.return
  }

  llvm.func @print_u16(%0: i16) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_uint_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<5 x i8>

    %1 = llvm.zext %0 : i16 to i32
    llvm.call @printf(%ptr, %1)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, i32) -> i32

    llvm.return
  }

  llvm.func @print_i16(%0: i16) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_int_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<5 x i8>

    %1 = llvm.sext %0 : i16 to i32
    llvm.call @printf(%ptr, %1)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, i32) -> i32

    llvm.return
  }

  llvm.func @print_u32(%0: i32) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_uint_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<5 x i8>

    llvm.call @printf(%ptr, %0)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, i32) -> i32

    llvm.return
  }

  llvm.func @print_i32(%0: i32) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_int_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<5 x i8>

    llvm.call @printf(%ptr, %0)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, i32) -> i32

    llvm.return
  }

  llvm.func @print_u64(%0: i64) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_ulong_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<6 x i8>

    llvm.call @printf(%ptr, %0)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, i64) -> i32

    llvm.return
  }

  llvm.func @print_i64(%0: i64) -> () {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %base = llvm.mlir.addressof @_long_fmt : !llvm.ptr<0>
    %ptr = llvm.getelementptr %base[%c0, %c0]
      : (!llvm.ptr<0>, i64, i64) -> !llvm.ptr<0>, !llvm.array<6 x i8>

    llvm.call @printf(%ptr, %0)
      { var_callee_type = !llvm.func<i32 (!llvm.ptr<0>, ...)> }
      : (!llvm.ptr<0>, i64) -> i32

    llvm.return
  }
}
