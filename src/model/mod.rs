//! The model half: what the trained artifact says about itself.
//!
//! Everything here reads the `.model` protobuf and nothing else — no tokenizer runtime, no
//! samples, no corpus. That is a narrower input than the guide's checklist assumes, and it turns
//! out to be a stronger one for most of it: a property read from the vocabulary or the trainer
//! spec holds for every possible input, where the same property tested by encoding samples holds
//! only for the samples.

pub mod artifact;
pub mod config;
pub mod pieces;
pub mod suite;
pub mod writing;
