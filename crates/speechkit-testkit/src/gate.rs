//! A latch a test holds closed to simulate a long native call.

use std::{
    sync::{Arc, Condvar, Mutex, PoisonError},
    time::{Duration, Instant},
};

#[derive(Default)]
struct State {
    released: bool,
    entered: usize,
}

/// A shared latch. Backend code calls [`wait`](Self::wait), which blocks
/// until the test calls [`release`](Self::release). Clones share the latch.
#[derive(Clone, Default)]
pub struct Gate {
    inner: Arc<(Mutex<State>, Condvar)>,
}

impl Gate {
    /// A closed gate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Opens the gate for every current and future waiter.
    pub fn release(&self) {
        let (state, changed) = &*self.inner;
        state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .released = true;
        changed.notify_all();
    }

    /// Blocks until the gate is released.
    pub fn wait(&self) {
        let (state, changed) = &*self.inner;
        let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
        state.entered += 1;
        changed.notify_all();
        while !state.released {
            state = changed.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// How many times [`wait`](Self::wait) was entered.
    pub fn entered(&self) -> usize {
        self.inner
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entered
    }

    /// Waits until [`wait`](Self::wait) was entered at least `count` times.
    /// Returns false on timeout.
    pub fn wait_entered(&self, count: usize, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let (state, changed) = &*self.inner;
        let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
        while state.entered < count {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            state = changed
                .wait_timeout(state, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
        true
    }
}

impl std::fmt::Debug for Gate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gate")
            .field("entered", &self.entered())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_opens_waiters() {
        let gate = Gate::new();
        let waiter = {
            let gate = gate.clone();
            std::thread::spawn(move || gate.wait())
        };
        assert!(gate.wait_entered(1, Duration::from_secs(5)));
        assert!(!waiter.is_finished());
        gate.release();
        waiter.join().unwrap();
        gate.wait();
        assert_eq!(gate.entered(), 2);
    }

    #[test]
    fn wait_entered_times_out() {
        assert!(!Gate::new().wait_entered(1, Duration::from_millis(10)));
    }
}
