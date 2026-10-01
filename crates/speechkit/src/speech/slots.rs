//! Execution slots: a counting semaphore that bounds concurrent sessions.

use std::{fmt, num::NonZeroUsize};

use super::sync::{Arc, Condvar, Mutex, lock};

/// Sessions an engine runs at once unless told otherwise (A-15, T-10).
pub(crate) const DEFAULT_MAX_SESSIONS: NonZeroUsize = match NonZeroUsize::new(8) {
    Some(max) => max,
    None => NonZeroUsize::MIN,
};

struct Inner {
    limit: usize,
    in_use: Mutex<usize>,
    /// Signalled when a slot is freed, or by [`SlotLimiter::wake`].
    freed: Condvar,
}

/// A counting semaphore built from a mutex.
///
/// Clones share the same slots. Each [`SlotGuard`] holds one slot until it
/// is dropped. The guard is `Send`, so a session's worker thread can own it
/// and release it only when the thread exits.
#[derive(Clone)]
pub(crate) struct SlotLimiter {
    inner: Arc<Inner>,
}

impl SlotLimiter {
    /// A limiter with `limit` slots.
    pub(crate) fn new(limit: NonZeroUsize) -> Self {
        Self {
            inner: Arc::new(Inner {
                limit: limit.get(),
                in_use: Mutex::new(0),
                freed: Condvar::new(),
            }),
        }
    }

    /// Takes a slot if one is free.
    pub(crate) fn try_acquire(&self) -> Option<SlotGuard> {
        let mut in_use = lock(&self.inner.in_use);
        if *in_use >= self.inner.limit {
            return None;
        }
        *in_use += 1;
        Some(self.guard())
    }

    /// Waits for a free slot and takes it, or returns `None` once
    /// `give_up` is true. `give_up` is checked under the limiter's lock
    /// before every wait, so a caller that makes it true and then calls
    /// [`wake`](Self::wake) is never missed.
    pub(crate) fn acquire(&self, give_up: impl Fn() -> bool) -> Option<SlotGuard> {
        let mut in_use = lock(&self.inner.in_use);
        loop {
            if give_up() {
                return None;
            }
            if *in_use < self.inner.limit {
                *in_use += 1;
                return Some(self.guard());
            }
            in_use = self
                .inner
                .freed
                .wait(in_use)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Wakes every [`acquire`](Self::acquire) so it checks `give_up` again.
    pub(crate) fn wake(&self) {
        let _in_use = lock(&self.inner.in_use);
        self.inner.freed.notify_all();
    }

    /// The number of slots currently held.
    pub(crate) fn in_use(&self) -> usize {
        *lock(&self.inner.in_use)
    }

    fn guard(&self) -> SlotGuard {
        SlotGuard {
            inner: self.inner.clone(),
        }
    }
}

impl fmt::Debug for SlotLimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlotLimiter")
            .field("limit", &self.inner.limit)
            .field("in_use", &self.in_use())
            .finish()
    }
}

/// One held slot. Dropping it frees the slot.
#[must_use = "dropping the guard frees the slot immediately"]
pub(crate) struct SlotGuard {
    inner: Arc<Inner>,
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        let mut in_use = lock(&self.inner.in_use);
        *in_use = in_use.saturating_sub(1);
        self.inner.freed.notify_all();
    }
}

impl fmt::Debug for SlotGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SlotGuard")
    }
}

#[cfg(test)]
#[cfg(not(speechkit_loom))]
mod tests {
    use super::*;

    fn two() -> SlotLimiter {
        SlotLimiter::new(NonZeroUsize::new(2).unwrap())
    }

    #[test]
    fn acquire_and_release() {
        let slots = two();
        let a = slots.try_acquire().unwrap();
        let b = slots.try_acquire().unwrap();
        assert_eq!(slots.in_use(), 2);
        assert!(slots.try_acquire().is_none());
        drop(a);
        assert_eq!(slots.in_use(), 1);
        let _c = slots.try_acquire().unwrap();
        drop(b);
        assert_eq!(slots.in_use(), 1);
    }

    #[test]
    fn acquire_waits_for_a_freed_slot_or_gives_up() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let slots = two();
        let a = slots.try_acquire().unwrap();
        let _b = slots.try_acquire().unwrap();
        let waiter = {
            let slots = slots.clone();
            std::thread::spawn(move || slots.acquire(|| false).is_some())
        };
        std::thread::sleep(std::time::Duration::from_millis(20));
        drop(a);
        assert!(waiter.join().unwrap());

        // Both slots are busy again, so this waiter can only give up.
        let _c = slots.try_acquire().unwrap();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let waiter = {
            let (slots, stop) = (slots.clone(), stop.clone());
            std::thread::spawn(move || slots.acquire(|| stop.load(Ordering::SeqCst)).is_none())
        };
        std::thread::sleep(std::time::Duration::from_millis(20));
        stop.store(true, Ordering::SeqCst);
        slots.wake();
        assert!(waiter.join().unwrap());
    }

    #[test]
    fn clones_share_slots() {
        let slots = two();
        let clone = slots.clone();
        let _a = slots.try_acquire().unwrap();
        let _b = clone.try_acquire().unwrap();
        assert!(slots.try_acquire().is_none());
        assert_eq!(clone.in_use(), 2);
    }
}

#[cfg(test)]
#[cfg(speechkit_loom)]
mod loom_tests {
    use loom::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn never_more_than_two_of_three() {
        loom::model(|| {
            let slots = SlotLimiter::new(NonZeroUsize::new(2).unwrap());
            let held = Arc::new(AtomicUsize::new(0));
            let handles: Vec<_> = (0..3)
                .map(|_| {
                    let slots = slots.clone();
                    let held = held.clone();
                    loom::thread::spawn(move || {
                        if let Some(guard) = slots.try_acquire() {
                            let now = held.fetch_add(1, Ordering::SeqCst) + 1;
                            assert!(now <= 2);
                            assert!(slots.in_use() <= 2);
                            held.fetch_sub(1, Ordering::SeqCst);
                            drop(guard);
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
            assert_eq!(slots.in_use(), 0);
        });
    }
}
