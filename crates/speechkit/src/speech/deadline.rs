//! Deadlines, and the helpers that make every blocking call handle them
//! the same way.

use std::time::{Duration, Instant};

use super::sync::{Condvar, MutexGuard};

/// When a waiting method gives up.
///
/// Every method that can wait takes `impl Into<Deadline>`:
///
/// - a [`Duration`] counts from the moment the call starts;
/// - an [`Instant`] is absolute, so several calls can share one budget.
///
/// ```
/// use std::time::{Duration, Instant};
///
/// use speechkit::Deadline;
///
/// let soon = Deadline::from(Duration::from_secs(5));
/// let shared = Deadline::from(Instant::now() + Duration::from_secs(5));
/// assert!(soon <= shared);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Deadline(Instant);

impl From<Instant> for Deadline {
    fn from(instant: Instant) -> Self {
        Self(instant)
    }
}

impl From<Duration> for Deadline {
    /// `duration` from now. A duration too long for an [`Instant`] gives
    /// the latest deadline the platform can represent near it.
    fn from(duration: Duration) -> Self {
        Self(instant_after(Instant::now(), duration))
    }
}

/// `start + duration`, or the latest instant the platform can represent
/// near it if the sum overflows. A caller's `Duration::MAX` therefore means
/// "never", where a plain `+` would panic.
pub(crate) fn instant_after(start: Instant, duration: Duration) -> Instant {
    let mut duration = duration;
    loop {
        if let Some(instant) = start.checked_add(duration) {
            return instant;
        }
        duration /= 2;
    }
}

impl Deadline {
    /// Time left, or `None` if the deadline has passed.
    pub(crate) fn remaining(self) -> Option<Duration> {
        let left = self.0.checked_duration_since(Instant::now())?;
        (!left.is_zero()).then_some(left)
    }
}

/// Waits on `condvar` until `ready` returns true or `deadline` passes.
///
/// `ready` is checked under the lock before every wait, so no wake-up is
/// lost. Returns the guard and whether `ready` became true.
pub(crate) fn wait_until<'a, T>(
    condvar: &Condvar,
    mut guard: MutexGuard<'a, T>,
    deadline: Deadline,
    mut ready: impl FnMut(&mut T) -> bool,
) -> (MutexGuard<'a, T>, bool) {
    loop {
        if ready(&mut guard) {
            return (guard, true);
        }
        let Some(left) = deadline.remaining() else {
            return (guard, false);
        };
        guard = match condvar.wait_timeout(guard, left) {
            Ok((guard, _)) => guard,
            Err(poisoned) => poisoned.into_inner().0,
        };
    }
}

/// Waits on `condvar` until `ready` returns true, with no deadline.
pub(crate) fn wait_forever<'a, T>(
    condvar: &Condvar,
    mut guard: MutexGuard<'a, T>,
    mut ready: impl FnMut(&mut T) -> bool,
) -> MutexGuard<'a, T> {
    while !ready(&mut guard) {
        guard = condvar
            .wait(guard)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    guard
}

#[cfg(test)]
#[cfg(not(speechkit_loom))]
mod tests {
    use std::{
        sync::{Arc, Condvar, Mutex},
        thread,
    };

    use super::*;

    #[test]
    fn remaining_is_none_once_passed() {
        assert!(Deadline::from(Instant::now()).remaining().is_none());
        assert!(Deadline::from(Duration::ZERO).remaining().is_none());
        let left = Deadline::from(Duration::from_secs(10)).remaining().unwrap();
        assert!(left > Duration::from_secs(9));
    }

    #[test]
    fn a_huge_duration_saturates() {
        let far = Deadline::from(Duration::MAX);
        assert!(far > Deadline::from(Duration::from_secs(1_000_000)));
    }

    #[test]
    fn instants_saturate_from_any_start() {
        let start = Instant::now();
        assert_eq!(
            instant_after(start, Duration::from_secs(3)),
            start + Duration::from_secs(3)
        );
        assert!(instant_after(start, Duration::MAX) > start + Duration::from_secs(1_000_000));
        assert_eq!(instant_after(start, Duration::ZERO), start);
    }

    #[test]
    fn wait_until_times_out() {
        let pair = (Mutex::new(false), Condvar::new());
        let start = Instant::now();
        let (_guard, ready) = wait_until(
            &pair.1,
            pair.0.lock().unwrap(),
            Deadline::from(start + Duration::from_millis(30)),
            |ready| *ready,
        );
        assert!(!ready);
        assert!(start.elapsed() >= Duration::from_millis(30));
    }

    #[test]
    fn wait_until_wakes_on_signal() {
        let pair = Arc::new((Mutex::new(false), Condvar::new()));
        let other = pair.clone();
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            *other.0.lock().unwrap() = true;
            other.1.notify_all();
        });
        let (guard, ready) = wait_until(
            &pair.1,
            pair.0.lock().unwrap(),
            Deadline::from(Duration::from_secs(10)),
            |ready| *ready,
        );
        assert!(ready);
        drop(guard);
        handle.join().unwrap();
        let guard = wait_forever(&pair.1, pair.0.lock().unwrap(), |ready| *ready);
        assert!(*guard);
    }
}
