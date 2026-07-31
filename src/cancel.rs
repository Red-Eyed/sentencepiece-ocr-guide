use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use thiserror::Error;

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
#[error("interrupted")]
pub struct Cancelled;

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn install_ctrlc_handler() -> Result<Self, ctrlc::Error> {
        let token = Self::default();
        let handler_token = token.clone();
        ctrlc::set_handler(move || {
            handler_token.cancel();
        })?;
        Ok(token)
    }

    #[cfg(test)]
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    #[cfg(not(test))]
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn check(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() {
            return Err(Cancelled);
        }
        Ok(())
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_cancellation_after_cancel() {
        let token = CancellationToken::default();

        assert!(token.check().is_ok());
        token.cancel();

        assert_eq!(token.check(), Err(Cancelled));
    }
}
