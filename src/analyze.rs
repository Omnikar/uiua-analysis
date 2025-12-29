use smallvec::SmallVec;
use std::collections::HashMap;
use uiua::Shape;

use crate::axis::Axis;

/// Symbolic shape
pub type SymShape = SmallVec<[Axis; 4]>;

pub enum ShapeInfo {
    Ranked(SymShape),
    Unranked { prefix: SymShape, suffix: SymShape },
}

pub struct Info {
    typ: u8,
    shape: ShapeInfo,
}

impl ShapeInfo {
    fn rank(&self) -> Option<usize> {
        match self {
            Self::Ranked(shape) => Some(shape.len()),
            _ => None,
        }
    }
}
