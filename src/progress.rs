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
}

pub struct StageProgress {
    bar: ProgressBar,
}

impl StageProgress {
    pub fn set_message(&self, message: impl Into<String>) {
        self.bar.set_message(message.into());
    }

    pub fn finish(self, message: impl Into<String>) {
        self.bar.finish_with_message(message.into());
    }
}
