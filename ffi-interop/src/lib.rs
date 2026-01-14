use std::{ffi, slice};

pub struct ArrayRef<'a, T> {
    pub rank: usize,
    pub shape: &'a [usize],
    pub strides: &'a [usize],
    pub data: &'a [T],
}

#[repr(C)]
pub struct FfiUnrankedMemRef {
    rank: usize,
    descriptor: *const ffi::c_void,
}

#[repr(C)]
struct FfiMemRefDescriptorHeader<T> {
    allocated: *const T,
    aligned: *const T,
    offset: usize,
    rest: ffi::c_void,
}

pub fn extract_memref<'a, T>(memref: FfiUnrankedMemRef) -> ArrayRef<'a, T> {
    let descr_ptr = memref.descriptor as *const FfiMemRefDescriptorHeader<T>;
    let descr = unsafe { &*descr_ptr };

    let shape_ptr = &descr.rest as *const ffi::c_void as *const usize;
    let shape = unsafe { slice::from_raw_parts(shape_ptr, memref.rank) };

    let strides_ptr = unsafe { shape_ptr.add(memref.rank) };
    let strides = unsafe { slice::from_raw_parts(strides_ptr, memref.rank) };

    let data_len = shape
        .iter()
        .zip(strides)
        .map(|(&len, &stride)| stride * (len - 1))
        .sum::<usize>()
        + 1;
    let data = unsafe { slice::from_raw_parts(descr.aligned.add(descr.offset), data_len) };

    ArrayRef {
        rank: memref.rank,
        shape,
        strides,
        data,
    }
}
