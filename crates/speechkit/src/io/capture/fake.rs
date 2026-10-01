//! The fake microphone the device contract tests drive.

use std::{
    sync::{Arc, Mutex, Weak, atomic::Ordering},
    time::Duration,
};

use ringbuf::{HeapProd, traits::Observer};

use super::{Device, lock};

/// What a fake microphone shares with the captures it starts.
#[derive(Default)]
pub(crate) struct Inner {
    /// The running capture, if any, and its ring buffer.
    current: Mutex<Option<(Weak<Device>, HeapProd<f32>)>>,
}

impl Inner {
    /// Makes `device`'s ring buffer the one [`FakeMicrophone::push`]
    /// fills.
    pub(super) fn attach(&self, device: &Arc<Device>, producer: HeapProd<f32>) {
        *lock(&self.current) = Some((Arc::downgrade(device), producer));
    }
}

/// The test's side of [`Microphone::fake`](super::Microphone::fake): it
/// delivers audio as a device callback would, and can lose the device.
/// For the device contract tests; not covered by semver.
#[doc(hidden)]
pub struct FakeMicrophone {
    pub(super) inner: Arc<Inner>,
}

impl FakeMicrophone {
    /// Delivers `samples` to the running capture, as device callbacks
    /// would, waiting while its ring buffer is full. Without a running
    /// capture, or once it stops, the rest is lost, as it would be.
    pub fn push(&self, samples: &[f32]) {
        let mut rest = samples;
        while !rest.is_empty() {
            let mut current = lock(&self.inner.current);
            let Some((device, producer)) = current.as_mut() else {
                return;
            };
            let Some(device) = device.upgrade() else {
                return;
            };
            if device.stopping.load(Ordering::Acquire) || device.device_lost() {
                return;
            }
            let room = producer.vacant_len().min(rest.len());
            if room == 0 {
                drop(current);
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            let (now, later) = rest.split_at(room);
            device.input(now, 1, producer);
            rest = later;
        }
    }

    /// Delivers `samples` in one device callback, without waiting for room
    /// in the ring buffer, as a device does while the capture thread has
    /// fallen behind: what does not fit is lost. Send more than the two
    /// seconds the buffer holds, so that some of it is lost however the
    /// capture thread is scheduled.
    pub fn push_burst(&self, samples: &[f32]) {
        let mut current = lock(&self.inner.current);
        let Some((device, producer)) = current.as_mut() else {
            return;
        };
        let Some(device) = device.upgrade() else {
            return;
        };
        if device.stopping.load(Ordering::Acquire) || device.device_lost() {
            return;
        }
        device.input(samples, 1, producer);
    }

    /// Loses the device, as a callback reporting an error would.
    pub fn lose(&self) {
        let current = lock(&self.inner.current);
        if let Some(device) = current.as_ref().and_then(|(device, _)| device.upgrade()) {
            device.lost.store(true, Ordering::Release);
        }
    }
}

impl std::fmt::Debug for FakeMicrophone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeMicrophone").finish_non_exhaustive()
    }
}
