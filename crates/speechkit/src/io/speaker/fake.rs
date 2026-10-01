//! The fake speaker the device contract tests drive.

use std::{
    sync::{Arc, Mutex, atomic::Ordering},
    time::Duration,
};

use ringbuf::HeapCons;

use super::{SKIPS, Shared, Skips, lock};

/// The test's side of [`Speaker::fake`](super::Speaker::fake): it plays
/// the speaker's audio when told to, as a device's callbacks would, and
/// can lose the device. For the device contract tests; not covered by
/// semver.
#[doc(hidden)]
pub struct FakeSpeaker {
    shared: Arc<Shared>,
    pub(super) consumer: Mutex<HeapCons<f32>>,
    /// The skip ranges the callback saw last, as a real one keeps them.
    skips: Mutex<Skips>,
}

impl FakeSpeaker {
    pub(super) fn new(shared: Arc<Shared>, consumer: HeapCons<f32>) -> Self {
        Self {
            shared,
            consumer: Mutex::new(consumer),
            skips: Mutex::new([0; 2 * SKIPS]),
        }
    }

    /// Plays `duration` of audio, as one device callback that has already
    /// been heard, and returns it, with silence where the speaker had
    /// nothing to play.
    pub fn play(&self, duration: Duration) -> Vec<f32> {
        let frames = usize::try_from(self.shared.rate.frames_in(duration)).unwrap_or(usize::MAX);
        let mut out = vec![0.0_f32; frames];
        let mut consumer = lock(&self.consumer);
        let mut skips = lock(&self.skips);
        // Its playback time is `duration` ago, so all of it counts as
        // played.
        self.shared
            .render_played(&mut *consumer, &mut out, duration, &mut skips);
        out
    }

    /// Loses the device, as a stream reporting an error would.
    pub fn lose(&self) {
        self.shared.lost.store(true, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for FakeSpeaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeSpeaker").finish_non_exhaustive()
    }
}
