use ffi_interop::{FfiUnrankedMemRef, extract_memref};

#[unsafe(no_mangle)]
pub extern "C" fn print_u8(memref: FfiUnrankedMemRef) {
    let array_ref = extract_memref::<u8>(memref);
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
