use std::sync::Mutex;
use std::time::{Duration, Instant};

use indicatif::{ProgressBar, ProgressStyle};

const STAGE_BAR_TEMPLATE: &str =
    "{bar:40.cyan/blue} {bytes}/{total_bytes} {elapsed_precise} eta {eta_precise} {msg}";
const PLAIN_STATUS_INTERVAL: Duration = Duration::from_secs(10);

pub struct ProgressReporter {
    json_output: bool,
}

impl ProgressReporter {
    pub fn new(json_output: bool) -> Self {
        Self { json_output }
    }

    pub fn stage(&self, message: impl Into<String>) -> StageProgress {
        let message = message.into();
        if self.json_output {
            eprintln!("{message}");
            return StageProgress::plain(message);
        }

        let bar = ProgressBar::new_spinner();
        bar.set_style(
            ProgressStyle::with_template("{spinner} {msg}")
                .expect("spinner progress template is static"),
        );
        bar.enable_steady_tick(std::time::Duration::from_millis(120));
        bar.set_message(message);
        StageProgress::interactive(bar)
    }

    pub fn stage_bar(&self, message: impl Into<String>, total: u64) -> StageProgress {
        let message = message.into();
        if self.json_output {
            eprintln!("{message}");
            return StageProgress::plain(message);
        }

        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::with_template(STAGE_BAR_TEMPLATE)
                .expect("bar progress template is static")
                .progress_chars("=> "),
        );
        bar.set_message(message);
        StageProgress::interactive(bar)
    }
}

pub struct StageProgress {
    bar: ProgressBar,
    plain_output: bool,
    last_plain_emit: Mutex<Option<Instant>>,
}

impl StageProgress {
    fn interactive(bar: ProgressBar) -> Self {
        Self {
            bar,
            plain_output: false,
            last_plain_emit: Mutex::new(None),
        }
    }

    fn plain(initial_message: String) -> Self {
        let now = Instant::now();
        Self {
            bar: ProgressBar::hidden(),
            plain_output: true,
            last_plain_emit: Mutex::new(Some(now)),
        }
        .with_initial_message(initial_message)
    }

    fn with_initial_message(self, initial_message: String) -> Self {
        self.bar.set_message(initial_message);
        self
    }
}

impl StageProgress {
    pub fn set_message(&self, message: impl Into<String>) {
        let message = message.into();
        self.bar.set_message(message.clone());
        self.emit_plain_status(message);
    }

    pub fn inc(&self, delta: u64) {
        self.bar.inc(delta);
    }

    pub fn finish(self, message: impl Into<String>) {
        let message = message.into();
        if self.plain_output {
            eprintln!("{message}");
        }
        self.bar.finish_with_message(message);
    }

    fn emit_plain_status(&self, message: String) {
        if !self.plain_output {
            return;
        }

        let now = Instant::now();
        let mut last_emit = self
            .last_plain_emit
            .lock()
            .expect("plain progress status lock should not be poisoned");
        if last_emit
            .as_ref()
            .is_some_and(|last| now.duration_since(*last) < PLAIN_STATUS_INTERVAL)
        {
            return;
        }

        eprintln!("{message}");
        *last_emit = Some(now);
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
        assert!(STAGE_BAR_TEMPLATE.contains("{bytes}/{total_bytes}"));
        assert!(STAGE_BAR_TEMPLATE.contains("eta {eta_precise}"));
        stage.inc(1);
        assert_eq!(stage.bar.position(), 1);
        stage.finish("done");
    }
}
