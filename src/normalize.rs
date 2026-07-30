use unicode_normalization::UnicodeNormalization;

use crate::config::{CanonicalizationConfig, SoftHyphenPolicy, StripRule, UnicodeForm};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizedLine {
    pub text: String,
    pub changed: bool,
}

pub fn canonicalize_line(input: &str, config: &CanonicalizationConfig) -> CanonicalizedLine {
    let mapped = map_configured_characters(input, config);
    let normalized = normalize_unicode(&mapped, config.unicode_form);
    CanonicalizedLine {
        changed: normalized != input,
        text: normalized,
    }
}

fn map_configured_characters(input: &str, config: &CanonicalizationConfig) -> String {
    let mut output = String::with_capacity(input.len());
    let last_index = input.chars().count().saturating_sub(1);

    for (index, character) in input.chars().enumerate() {
        if should_strip(character, config) {
            continue;
        }

        if character == '\u{00ad}' {
            apply_soft_hyphen(&mut output, index == last_index, config.soft_hyphen);
            continue;
        }

        if character == '\u{00a0}' && config.map_nbsp_to_space {
            output.push(' ');
            continue;
        }

        if is_arabic_presentation_form(character) && config.fold_arabic_presentation_forms {
            output.push_str(&character.to_string().nfkc().collect::<String>());
            continue;
        }

        output.push(character);
    }

    output
}

fn should_strip(character: char, config: &CanonicalizationConfig) -> bool {
    config.strip.iter().any(|rule| match rule {
        StripRule::Bom => character == '\u{feff}',
        StripRule::ZeroWidthSpace => character == '\u{200b}',
    })
}

fn apply_soft_hyphen(output: &mut String, is_line_final: bool, policy: SoftHyphenPolicy) {
    match policy {
        SoftHyphenPolicy::LineFinalToHyphenMidlineStrip => {
            if is_line_final {
                output.push('-');
            }
        }
    }
}

fn is_arabic_presentation_form(character: char) -> bool {
    let codepoint = character as u32;
    (0xfb50..=0xfdff).contains(&codepoint) || (0xfe70..=0xfeff).contains(&codepoint)
}

fn normalize_unicode(input: &str, form: UnicodeForm) -> String {
    match form {
        UnicodeForm::Nfc => input.nfc().collect(),
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{CanonicalizationConfig, SoftHyphenPolicy, StripRule, UnicodeForm};

    use super::*;

    fn config() -> CanonicalizationConfig {
        CanonicalizationConfig {
            unicode_form: UnicodeForm::Nfc,
            strip: vec![StripRule::Bom, StripRule::ZeroWidthSpace],
            map_nbsp_to_space: true,
            fold_arabic_presentation_forms: true,
            soft_hyphen: SoftHyphenPolicy::LineFinalToHyphenMidlineStrip,
            preserve_zwj_zwnj: true,
            preserve_compatibility_chars: true,
        }
    }

    #[test]
    fn composes_nfc_and_maps_configured_artifacts() {
        let input = "\u{feff}cafe\u{0301}\u{00a0}\u{200b}";

        let line = canonicalize_line(input, &config());

        assert_eq!(line.text, "café ");
        assert!(line.changed);
    }

    #[test]
    fn preserves_compatibility_characters_and_joiners() {
        let input = "Ａ ﬁ \u{200c}\u{200d}";

        let line = canonicalize_line(input, &config());

        assert_eq!(line.text, input);
        assert!(!line.changed);
    }

    #[test]
    fn folds_arabic_presentation_forms_only() {
        let line = canonicalize_line("\u{fedb}", &config());

        assert_eq!(line.text, "ك");
    }

    #[test]
    fn maps_only_line_final_soft_hyphen_to_visible_hyphen() {
        let config = config();

        assert_eq!(
            canonicalize_line("co\u{00ad}operate", &config).text,
            "cooperate"
        );
        assert_eq!(canonicalize_line("line\u{00ad}", &config).text, "line-");
    }
}
