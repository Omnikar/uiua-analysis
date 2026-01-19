use crate::{ArrayRef, FfiUnrankedMemRef, extract_memref};
use concat_idents::concat_idents;

fn show_num<T: std::fmt::Display>(arr: ArrayRef<T>) {
    if arr.rank == 0 {
        println!("{}", arr.data[0])
    } else if arr.rank == 1 {
        print!("[");
        if !arr.data.is_empty() {
            print!("{}", arr.data[0]);
            for num in &arr.data[1..] {
                print!(" {num}");
            }
        }
        println!("]");
    } else {
        let breakvals = arr
            .shape
            .iter()
            .rev()
            .skip(1)
            .copied()
            .fold(vec![1], |mut acc, next| {
                acc.push(*acc.last().unwrap() * next);
                acc
            });
        let str_data = (0..arr.elem_count())
            .map(|i| arr[i].to_string())
            .collect::<Vec<String>>();
        let longest = str_data.iter().map(String::len).max().unwrap_or(0);
        let str_data = str_data
            .into_iter()
            .map(|x| format!("{x:>0$}", longest))
            .collect::<Vec<String>>();
        print!("╭─ ");
        print!("{}", arr.shape[0]);
        for axis in &arr.shape[1..] {
            print!("×{axis}");
        }
        print!(" {}", std::any::type_name::<T>());
        for (i, row) in str_data.chunks(*arr.shape.last().unwrap()).enumerate() {
            for &val in &breakvals {
                if i % val == 0 && (i != 0 || val == 1) {
                    println!()
                }
            }
            print!(" ");
            for num in row {
                print!(" {num}");
            }
        }
        println!("\n╰─ ");
    }
}

macro_rules! create_ffi_print_funcs {
    ($($t:ty),+) => {
        $(
            concat_idents!(fn_name = print_show_, $t {
                /// # Safety
                /// The given struct must be a valid MLIR unranked memref
                #[unsafe(no_mangle)]
                pub unsafe extern "C" fn fn_name(memref: FfiUnrankedMemRef) {
                    let array_ref = unsafe { extract_memref::<$t>(memref) };
                    show_num(array_ref);
                }
            });
        )+
    };
}

create_ffi_print_funcs!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

/// # Safety
/// The given struct must be a valid MLIR unranked memref
#[unsafe(no_mangle)]
pub unsafe extern "C" fn print_show_bool(memref: FfiUnrankedMemRef) {
    let array_ref = unsafe { extract_memref::<u8>(memref) };
    show_num(array_ref);
}
