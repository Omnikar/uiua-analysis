use ffi_interop::{extract_memref, FfiUnrankedMemRef};

/// # Safety
/// The given struct must be a valid MLIR unranked memref
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pretty_print_show_u8(memref: FfiUnrankedMemRef) {
    let array_ref = unsafe { extract_memref::<u8>(memref) };
    println!("Array: ");
    for num in array_ref.data {
        print!(" {num} ");
    }
    println!("\nShape: ");
    for axis in array_ref.shape {
        print!(" {axis} ");
    }
    println!("\nStride: ");
    for stride in array_ref.strides {
        print!(" {stride} ");
    }
    println!();
}
