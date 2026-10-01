//! What a speaker plays: a synthesis, a buffer, or samples pushed into a
//! sink.

use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex},
    time::Duration,
};

use super::lock;
use crate::{
    AudioBuffer, Deadline, SampleRate, SpeechError,
    speech::{audio::find_invalid, deadline::wait_until},
    tts::{PushedText, TtsOutput},
};

/// How much audio a [`Sink`] holds before `push` waits.
const SINK_QUEUE: Duration = Duration::from_secs(2);

/// Something a [`Speaker`](super::Speaker) plays: a synthesis's
/// [`TtsOutput`] or an [`AudioBuffer`].
pub struct Sound(pub(super) Source);

pub(super) enum Source {
    Tts(TtsOutput),
    Buffer(AudioBuffer),
    Sink(Arc<Channel>),
}

impl Sound {
    /// The text pushed into the synthesis, if the sound is one.
    pub(super) fn text(&self) -> Option<PushedText> {
        match &self.0 {
            Source::Tts(output) => Some(output.pushed_text()),
            Source::Buffer(_) | Source::Sink(_) => None,
        }
    }

    pub(super) fn sample_rate(&self) -> SampleRate {
        match &self.0 {
            Source::Tts(output) => output.sample_rate(),
            Source::Buffer(buffer) => buffer.sample_rate,
            Source::Sink(channel) => channel.rate,
        }
    }
}

impl From<TtsOutput> for Sound {
    fn from(output: TtsOutput) -> Self {
        Self(Source::Tts(output))
    }
}

impl From<AudioBuffer> for Sound {
    fn from(buffer: AudioBuffer) -> Self {
        Self(Source::Buffer(buffer))
    }
}

impl std::fmt::Debug for Sound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match &self.0 {
            Source::Tts(_) => "synthesis",
            Source::Buffer(_) => "buffer",
            Source::Sink(_) => "sink",
        };
        f.debug_struct("Sound")
            .field("kind", &kind)
            .field("sample_rate", &self.sample_rate())
            .finish()
    }
}

/// The queue between a [`Sink`] and the player.
pub(super) struct Channel {
    rate: SampleRate,
    capacity: usize,
    state: Mutex<ChannelState>,
    changed: Condvar,
}

struct ChannelState {
    samples: VecDeque<f32>,
    /// The sink was closed: the playback ends after what was pushed.
    closed: bool,
    /// The playback ended: pushes fail.
    ended: bool,
}

/// What [`Channel::take`] found.
pub(super) enum Taken {
    Audio(Vec<f32>),
    Nothing,
    End,
}

impl Channel {
    pub(super) fn new(rate: SampleRate) -> Self {
        Self {
            rate,
            capacity: usize::try_from(rate.frames_in(SINK_QUEUE))
                .unwrap_or(96_000)
                .max(1),
            state: Mutex::new(ChannelState {
                samples: VecDeque::new(),
                closed: false,
                ended: false,
            }),
            changed: Condvar::new(),
        }
    }

    /// Takes up to `max` samples, waiting until `deadline` for some.
    pub(super) fn take(&self, max: usize, deadline: Deadline) -> Taken {
        let (mut state, _) = wait_until(&self.changed, lock(&self.state), deadline, |state| {
            !state.samples.is_empty() || state.closed
        });
        if state.samples.is_empty() {
            return if state.closed {
                Taken::End
            } else {
                Taken::Nothing
            };
        }
        let count = max.min(state.samples.len());
        let taken = state.samples.drain(..count).collect();
        drop(state);
        self.changed.notify_all();
        Taken::Audio(taken)
    }

    /// Refuses later pushes: the playback ended or was stopped.
    pub(super) fn end(&self) {
        let mut state = lock(&self.state);
        state.ended = true;
        state.samples.clear();
        drop(state);
        self.changed.notify_all();
    }

    fn close(&self) {
        lock(&self.state).closed = true;
        self.changed.notify_all();
    }
}

/// Samples you push yourself, from [`Speaker::sink`](super::Speaker::sink).
///
/// Its [`Playback`](super::Playback) plays them as they come, resampled to
/// the device's rate, and ends after [`close`](Self::close). The sink
/// holds up to 2 s of audio; [`push`](Self::push) waits while it is full.
/// Dropping it closes it.
pub struct Sink {
    channel: Arc<Channel>,
}

impl Sink {
    pub(super) fn new(channel: Arc<Channel>) -> Self {
        Self { channel }
    }

    /// The rate of the samples it takes.
    pub fn sample_rate(&self) -> SampleRate {
        self.channel.rate
    }

    /// Queues mono `samples` at the sink's rate, waiting while the queue is
    /// full until `deadline`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for a sample that is not finite or is
    /// outside [-1.0, 1.0], and nothing is queued;
    /// [`SpeechError::DeadlineExceeded`] if the deadline passes with part
    /// of them not queued; [`SpeechError::Closed`] after
    /// [`close`](Self::close), or once the playback was stopped or ended.
    pub fn push(&self, samples: &[f32], deadline: impl Into<Deadline>) -> Result<(), SpeechError> {
        if let Some(bad) = find_invalid(samples) {
            return Err(bad.into());
        }
        let deadline = deadline.into();
        let channel = &*self.channel;
        let mut rest = samples;
        while !rest.is_empty() {
            let (mut state, _) =
                wait_until(&channel.changed, lock(&channel.state), deadline, |state| {
                    state.ended || state.closed || state.samples.len() < channel.capacity
                });
            if state.ended || state.closed {
                return Err(SpeechError::Closed);
            }
            let room = channel.capacity.saturating_sub(state.samples.len());
            if room == 0 {
                return Err(SpeechError::DeadlineExceeded);
            }
            let (now, later) = rest.split_at(room.min(rest.len()));
            state.samples.extend(now);
            rest = later;
            drop(state);
            channel.changed.notify_all();
        }
        Ok(())
    }

    /// Ends the sound: its playback ends after what was pushed.
    pub fn close(&self) {
        self.channel.close();
    }
}

impl Drop for Sink {
    fn drop(&mut self) {
        self.close();
    }
}

impl std::fmt::Debug for Sink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sink")
            .field("sample_rate", &self.channel.rate)
            .finish_non_exhaustive()
    }
}
