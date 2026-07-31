use indicatif::{ProgressBar, ProgressStyle};

pub struct ProgressReporter {
    json_output: bool,
}

impl ProgressReporter {
    pub fn new(json_output: bool) -> Self {
        Self { json_output }
    }

    pub fn stage(&self, message: &'static str) -> StageProgress {
        if self.json_output {
            eprintln!("{message}");
            return StageProgress {
                bar: ProgressBar::hidden(),
            };
        }

        let bar = ProgressBar::new_spinner();
        bar.set_style(
            ProgressStyle::with_template("{spinner} {msg}")
                .expect("spinner progress template is static"),
        );
        bar.enable_steady_tick(std::time::Duration::from_millis(120));
        bar.set_message(message);
        StageProgress { bar }
    }

    pub fn stage_bar(&self, message: &'static str, total: u64) -> StageProgress {
        if self.json_output {
            eprintln!("{message}");
            return StageProgress {
                bar: ProgressBar::hidden(),
            };
        }

        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::with_template("{bar:40.cyan/blue} {pos}/{len} {msg}")
                .expect("bar progress template is static")
                .progress_chars("=> "),
        );
        bar.set_message(message);
        StageProgress { bar }
    }
}

pub struct StageProgress {
    bar: ProgressBar,
}

impl StageProgress {
    pub fn set_message(&self, message: impl Into<String>) {
        self.bar.set_message(message.into());
    }

    pub fn inc(&self, delta: u64) {
        self.bar.inc(delta);
    }

    pub fn finish(self, message: impl Into<String>) {
        self.bar.finish_with_message(message.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_bar_tracks_position() {
        let progress = ProgressReporter::new(false);
        let stage = progress.stage_bar("fixing corpus", 3);

        assert_eq!(stage.bar.length(), Some(3));
        stage.inc(1);
        assert_eq!(stage.bar.position(), 1);
        stage.finish("done");
    }
}
