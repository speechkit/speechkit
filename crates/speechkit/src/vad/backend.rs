//! `VadBackend`: a VAD in front of an offline recognizer.

use std::{sync::Arc, time::Duration};

use super::{
    OfflineRecognizer, VadModel,
    tracker::{Detected, Tracker},
};
use crate::{
    Flow, SpeechError,
    asr::{
        AsrBackend, AsrCapabilities, AsrEvent, AsrEvents, AsrOptions, AsrStream, Segment,
        UtteranceId,
    },
};

/// The default longest utterance before it is cut.
const DEFAULT_MAX_UTTERANCE: Duration = Duration::from_secs(20);

/// Runs an [`OfflineRecognizer`] on each utterance a [`Vad`](super::Vad) finds.
///
/// Each session gets its own VAD, which also reports speech activity:
///
/// - `SpeechStarted` where the VAD says speech started;
/// - `SpeechEnded` at the end of each segment after which speech stopped;
/// - `ActivityKnown` after each call: while speech goes on, its start;
///   otherwise the audio fed minus the model's
///   [`start_delay`](VadModel::start_delay).
///
/// Each speech segment becomes one [`Segment`], timed by the VAD, with
/// utterance IDs counting up from 0. A segment with no words is still sent,
/// as the contract asks, and so is an empty one for speech the VAD dropped
/// as too short. An utterance longer than
/// [`with_max_utterance`](Self::with_max_utterance) is cut and committed
/// without `SpeechEnded`, since the speech goes on. There are never
/// partial results.
pub struct VadBackend<R, V> {
    recognizer: Arc<R>,
    vad: V,
    name: String,
    caps: AsrCapabilities,
    max_utterance: Duration,
}

impl<R: OfflineRecognizer, V: VadModel> VadBackend<R, V> {
    /// Combines `recognizer` with detectors made by `vad`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] if the VAD's rate differs from the
    /// recognizer's preferred rate.
    pub fn new(recognizer: R, vad: V) -> Result<Self, SpeechError> {
        let mut caps = recognizer.capabilities().clone();
        if vad.sample_rate() != caps.sample_rate {
            return Err(SpeechError::InvalidInput(format!(
                "the VAD runs at {} but the recognizer wants {}",
                vad.sample_rate(),
                caps.sample_rate
            )));
        }
        caps.reports_partials = false;
        caps.reports_activity = true;
        let name = format!("{}+vad", recognizer.name());
        Ok(Self {
            recognizer: Arc::new(recognizer),
            vad,
            name,
            caps,
            max_utterance: DEFAULT_MAX_UTTERANCE,
        })
    }

    /// Cuts an utterance that runs this long and commits it. Default: 20 s.
    /// A VAD that cannot honor it fails each session at `start`.
    #[must_use]
    pub fn with_max_utterance(mut self, max_utterance: Duration) -> Self {
        self.max_utterance = max_utterance;
        self
    }
}

impl<R: OfflineRecognizer, V: VadModel> AsrBackend for VadBackend<R, V> {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn open(
        &self,
        opts: &AsrOptions,
        events: AsrEvents,
    ) -> Result<Box<dyn AsrStream>, SpeechError> {
        let vad = self.vad.create(self.max_utterance)?;
        Ok(Box::new(VadStream {
            events,
            tracker: Tracker::new(vad, self.caps.sample_rate, self.vad.start_delay()),
            recognizer: self.recognizer.clone(),
            opts: opts.clone(),
            next: 0,
        }))
    }
}

impl<R, V> std::fmt::Debug for VadBackend<R, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VadBackend")
            .field("name", &self.name)
            .field("max_utterance", &self.max_utterance)
            .finish_non_exhaustive()
    }
}

struct VadStream<R> {
    events: AsrEvents,
    tracker: Tracker,
    recognizer: Arc<R>,
    opts: AsrOptions,
    /// The next utterance ID.
    next: u64,
}

/// Stops at the first [`Flow::Stop`].
macro_rules! send {
    ($self:ident, $event:expr) => {
        if $self.events.send($event) == Flow::Stop {
            return Ok(());
        }
    };
}

impl<R: OfflineRecognizer> VadStream<R> {
    /// Sends the events for what the tracker found.
    fn report(&mut self, detected: Vec<Detected>) -> Result<(), SpeechError> {
        for found in detected {
            match found {
                Detected::Started(at) => send!(self, AsrEvent::SpeechStarted { at }),
                Detected::Segment {
                    segment,
                    end,
                    ended,
                } => {
                    let utterance = self.utterance();
                    if ended {
                        send!(self, AsrEvent::SpeechEnded { at: end, utterance });
                    }
                    let text = if segment.samples.is_empty() {
                        String::new()
                    } else {
                        self.recognizer.recognize(&segment.samples, &self.opts)?
                    };
                    send!(
                        self,
                        AsrEvent::Segment(Segment {
                            utterance,
                            text,
                            start: segment.start,
                            end,
                        })
                    );
                }
                Detected::Dropped { start, end } => {
                    let utterance = self.utterance();
                    send!(self, AsrEvent::SpeechEnded { at: end, utterance });
                    send!(
                        self,
                        AsrEvent::Segment(Segment {
                            utterance,
                            text: String::new(),
                            start,
                            end,
                        })
                    );
                }
                Detected::Known(through) => send!(self, AsrEvent::ActivityKnown { through }),
            }
        }
        Ok(())
    }

    fn utterance(&mut self) -> UtteranceId {
        let id = UtteranceId(self.next);
        self.next += 1;
        id
    }
}

impl<R: OfflineRecognizer> AsrStream for VadStream<R> {
    fn accept(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        let detected = self.tracker.accept(samples);
        self.report(detected)
    }

    fn finish(&mut self) -> Result<(), SpeechError> {
        let detected = self.tracker.finish();
        self.report(detected)
    }
}
