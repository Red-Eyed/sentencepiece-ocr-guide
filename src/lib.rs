//! Validate a SentencePiece tokenizer for multilingual OCR, and the corpus behind it.
//!
//! Two checklists over one vocabulary of findings:
//!
//! - **corpus**, before training — which encoding axes actually vary, and in which source.
//! - **model**, after training — what the trained artifact says about itself.
//!
//! The order is not cosmetic. Several model defects originate in the corpus, so a model failure
//! can carry [`report::Remedy::FixCorpus`]: retraining alone would reproduce it.

pub mod corpus;
pub mod crosscheck;
pub mod model;
pub mod render;
pub mod report;
pub mod writing;
