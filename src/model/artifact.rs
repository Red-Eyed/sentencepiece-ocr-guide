//! The trained model, read once into plain data.
//!
//! Everything the model half checks comes from the `.model` protobuf: the piece inventory, and
//! the trainer and normalizer settings the run was configured with. Parsing happens here and
//! produces owned, ordinary structs, which is what lets every check downstream be a pure
//! function over data a test can build by hand.
//!
//! Reading the *settings* is the part `SentencePieceProcessor` cannot do at all — the runtime
//! API exposes the vocabulary but not the flags it was trained with. Several checks that are
//! only inferrable by experiment through that API are plain facts here.

use std::path::Path;

use sentencepiece_model::{ModelType, SentencePieceModel, Type};

/// What a piece is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
    /// Present in the file but never emitted.
    Unused,
}

impl Kind {
    fn from_proto(kind: Type) -> Self {
        match kind {
            Type::Normal => Kind::Normal,
            Type::Unknown => Kind::Unknown,
            Type::Control => Kind::Control,
            Type::UserDefined => Kind::UserDefined,
            Type::Byte => Kind::Byte,
            Type::Unused => Kind::Unused,
        }
    }

    /// Whether this piece carries surface text worth inspecting.
    ///
    /// `<s>`, `<unk>` and `<0x41>` are machinery rather than text — a check for digit-only or
    /// cross-script pieces that included them would report on its own scaffolding. `Unused` is
    /// excluded too: it cannot be emitted, so a defect in one is unreachable.
    fn is_inspectable(self) -> bool {
        matches!(self, Kind::Normal | Kind::UserDefined)
    }
}

/// One vocabulary entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    pub text: String,
    pub kind: Kind,
}

impl Piece {
    pub fn new(text: impl Into<String>, kind: Kind) -> Self {
        Self {
            text: text.into(),
            kind,
        }
    }
}

/// SentencePiece writes a word-initial space as U+2581 rather than a literal space.
pub const SPACE_MARKER: char = '▁';

/// A piece's text with the space marker turned back into the space it stands for.
///
/// Checks that read what a piece *says* need this; the one check that cares whether the marker
/// is present reads the raw text instead.
pub fn surface(text: &str) -> String {
    text.replace(SPACE_MARKER, " ")
}

/// The piece inventory.
#[derive(Debug, Clone, Default)]
pub struct Vocabulary {
    pieces: Vec<Piece>,
}

impl Vocabulary {
    pub fn new(pieces: Vec<Piece>) -> Self {
        Self { pieces }
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// The pieces whose text is worth checking — ordinary and user-defined entries.
    pub fn inspectable(&self) -> impl Iterator<Item = &Piece> {
        self.pieces.iter().filter(|p| p.kind.is_inspectable())
    }

    pub fn count_of(&self, kind: Kind) -> usize {
        self.pieces.iter().filter(|p| p.kind == kind).count()
    }

    pub fn has_user_defined(&self, text: &str) -> bool {
        self.pieces
            .iter()
            .any(|p| p.kind == Kind::UserDefined && p.text == text)
    }
}

/// The trainer settings the model records about its own training run.
#[derive(Debug, Clone)]
pub struct Trainer {
    pub model_type: ModelType,
    pub vocab_size: i32,
    pub byte_fallback: bool,
    pub split_digits: bool,
    pub split_by_unicode_script: bool,
    pub max_piece_length: i32,
    pub character_coverage: f32,
    pub user_defined_symbols: Vec<String>,
}

/// The normalizer settings, which decide what happens to text before it is ever tokenized.
#[derive(Debug, Clone)]
pub struct Normalizer {
    pub rule_name: String,
    /// A compiled character-folding table. Empty is what `identity` produces.
    pub has_charsmap: bool,
    pub add_dummy_prefix: bool,
    pub remove_extra_whitespaces: bool,
}

/// A parsed `.model` file.
///
/// The two spec halves are optional because the protobuf makes them optional. A model without
/// them is readable but unjudgeable on configuration, and the suite skips those checks with the
/// reason rather than assuming defaults — an assumed default would be a verdict on a setting
/// nobody recorded.
#[derive(Debug, Clone)]
pub struct Artifact {
    pub vocabulary: Vocabulary,
    pub trainer: Option<Trainer>,
    pub normalizer: Option<Normalizer>,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("{path}: {source}")]
    Unreadable {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// Read a `.model` file into plain data. The only function here that touches disk.
pub fn load(path: &Path) -> Result<Artifact, LoadError> {
    let model = SentencePieceModel::from_file(path).map_err(|source| LoadError::Unreadable {
        path: path.display().to_string(),
        source,
    })?;

    Ok(Artifact {
        vocabulary: read_vocabulary(&model),
        trainer: model.trainer().map(read_trainer),
        normalizer: model.normalizer().map(read_normalizer),
    })
}

fn read_vocabulary(model: &SentencePieceModel) -> Vocabulary {
    let pieces = model
        .pieces()
        .iter()
        .map(|piece| Piece {
            text: piece.piece().to_string(),
            kind: Kind::from_proto(piece.r#type()),
        })
        .collect();
    Vocabulary::new(pieces)
}

fn read_trainer(spec: &sentencepiece_model::TrainerSpec) -> Trainer {
    Trainer {
        model_type: spec.model_type(),
        vocab_size: spec.vocab_size(),
        byte_fallback: spec.byte_fallback(),
        split_digits: spec.split_digits(),
        split_by_unicode_script: spec.split_by_unicode_script(),
        max_piece_length: spec.max_sentencepiece_length(),
        character_coverage: spec.character_coverage(),
        user_defined_symbols: spec.user_defined_symbols.clone(),
    }
}

fn read_normalizer(spec: &sentencepiece_model::NormalizerSpec) -> Normalizer {
    Normalizer {
        rule_name: spec.name().to_string(),
        has_charsmap: !spec.precompiled_charsmap().is_empty(),
        add_dummy_prefix: spec.add_dummy_prefix(),
        remove_extra_whitespaces: spec.remove_extra_whitespaces(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_space_marker_reads_back_as_a_space() {
        assert_eq!(surface("▁hello"), " hello");
        assert_eq!(surface("plain"), "plain");
    }

    #[test]
    fn machinery_pieces_are_not_inspectable() {
        // A check for digit-only pieces must not report on `<0x30>`.
        let vocabulary = Vocabulary::new(vec![
            Piece::new("word", Kind::Normal),
            Piece::new("\\frac", Kind::UserDefined),
            Piece::new("<0x30>", Kind::Byte),
            Piece::new("<s>", Kind::Control),
            Piece::new("<unk>", Kind::Unknown),
            Piece::new("stale", Kind::Unused),
        ]);

        let inspected: Vec<&str> = vocabulary.inspectable().map(|p| p.text.as_str()).collect();
        assert_eq!(inspected, vec!["word", "\\frac"]);
    }

    #[test]
    fn user_defined_symbols_are_found_by_kind_not_by_text() {
        // A piece that merely happens to spell the symbol is not the same as a declared one.
        let vocabulary = Vocabulary::new(vec![
            Piece::new("\\frac", Kind::UserDefined),
            Piece::new("\\sum", Kind::Normal),
        ]);
        assert!(vocabulary.has_user_defined("\\frac"));
        assert!(!vocabulary.has_user_defined("\\sum"));
    }

    #[test]
    fn byte_pieces_are_counted_by_kind() {
        let vocabulary = Vocabulary::new(vec![
            Piece::new("<0x00>", Kind::Byte),
            Piece::new("<0x01>", Kind::Byte),
            Piece::new("word", Kind::Normal),
        ]);
        assert_eq!(vocabulary.count_of(Kind::Byte), 2);
    }
}
