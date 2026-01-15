mod print;

pub use print::*;

use std::{ffi, slice};

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

struct ArrayRef<'a, T> {
    rank: usize,
    shape: &'a [usize],
    strides: &'a [usize],
    data: &'a [T],
}

/// Create an `ArrayRef` instance out of a raw MLIR unranked memref
/// # Safety
/// This operation requires the given struct to be a valid MLIR unranked memref
unsafe fn extract_memref<'a, T>(memref: FfiUnrankedMemRef) -> ArrayRef<'a, T> {
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

impl<'a, T> std::ops::Index<&[usize]> for ArrayRef<'a, T> {
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let flatindex = index
            .iter()
            .zip(self.strides)
            .map(|(&a, &b)| a * b)
            .sum::<usize>();
        &self.data[flatindex]
    }
}

impl<'a, T> std::ops::Index<usize> for ArrayRef<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self[&*base_conv(index, self.shape)]
    }
}

impl<'a, T> ArrayRef<'a, T> {
    fn elem_count(&self) -> usize {
        self.shape.iter().copied().product()
    }
}

fn base_conv(index: usize, bases: &[usize]) -> Vec<usize> {
    let mut out = bases
        .iter()
        .copied()
        .rev()
        .fold((index, Vec::new()), |(rem, mut digits), base| {
            digits.push(rem % base);
            (rem / base, digits)
        })
        .1;
    out.reverse();
    out
}
