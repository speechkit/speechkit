//! The DashScope recognition task state machine.

use std::{collections::HashSet, time::Duration};

use crate::{
    SampleRate, SpeechError,
    asr::{AsrEvent, Partial, Segment, UtteranceId},
};

use super::protocol::{BACKEND, Event, Sentence};

/// Where a task is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Waiting for `task-started`.
    Starting,
    /// Audio is flowing.
    Running,
    /// `finish-task` was sent.
    Finishing,
    /// `task-finished` arrived.
    Finished,
}

/// Turns task events into backend events.
///
/// Partial sentences update the current utterance; a final sentence
/// commits it and moves to the next. A final sentence repeated with the
/// same identity (ID and time range) is dropped; one with no identity at
/// all always counts, since a repeated phrase is legitimate.
///
/// Sentences are the speech: a sentence's first result starts speech at
/// its `begin_time`, and its final result ends it at its `end_time`.
/// Between sentences, activity is known up to the last end, or the audio
/// sent less `lag`, whichever is later.
#[derive(Debug)]
pub(crate) struct DashScopeState {
    phase: Phase,
    utterance: u64,
    seen: HashSet<String>,
    lag: Duration,
    /// Audio sent, at 16 kHz.
    frames: u64,
    /// The start of the speech in progress.
    speech: Option<Duration>,
    /// Where the last sentence ended.
    last_stop: Duration,
    /// The last `ActivityKnown` sent.
    known: Duration,
}

impl Default for DashScopeState {
    fn default() -> Self {
        Self::new(Duration::from_secs(1))
    }
}

fn millis(value: f64) -> Duration {
    Duration::try_from_secs_f64(value / 1_000.0).unwrap_or_default()
}

impl DashScopeState {
    /// A task that is starting, whose service reports speech at most
    /// `lag` behind the audio sent.
    pub(crate) fn new(lag: Duration) -> Self {
        Self {
            phase: Phase::Starting,
            utterance: 0,
            seen: HashSet::new(),
            lag,
            frames: 0,
            speech: None,
            last_stop: Duration::ZERO,
            known: Duration::ZERO,
        }
    }

    /// Records audio sent, and returns what that says about activity.
    pub(crate) fn sent(&mut self, frames: usize) -> Vec<AsrEvent> {
        self.frames += frames as u64;
        let mut out = Vec::new();
        self.report_activity(&mut out);
        out
    }

    fn sent_time(&self) -> Duration {
        SampleRate::HZ_16000.duration_of(self.frames)
    }

    /// The task finished: activity is known to the end of the audio.
    pub(crate) fn finished(&mut self) -> Vec<AsrEvent> {
        let through = self.sent_time();
        if self.speech.is_none() && through > self.known {
            self.known = through;
            return vec![AsrEvent::ActivityKnown { through }];
        }
        Vec::new()
    }

    fn report_activity(&mut self, out: &mut Vec<AsrEvent>) {
        let candidate = self
            .sent_time()
            .saturating_sub(self.lag)
            .max(self.last_stop);
        let through = self.speech.map_or(candidate, |start| candidate.min(start));
        if through > self.known {
            self.known = through;
            out.push(AsrEvent::ActivityKnown { through });
        }
    }

    /// Starts speech at `begin`, or where activity is known if the
    /// sentence has no time.
    fn start(&mut self, begin: Option<f64>, out: &mut Vec<AsrEvent>) {
        if self.speech.is_none() {
            let at = begin.map_or(self.known, millis);
            self.speech = Some(at);
            out.push(AsrEvent::SpeechStarted { at });
        }
    }

    /// The current phase.
    pub(crate) fn phase(&self) -> Phase {
        self.phase
    }

    /// `finish-task` was sent.
    pub(crate) fn finishing(&mut self) {
        self.phase = Phase::Finishing;
    }

    fn identity(sentence: &Sentence) -> Option<String> {
        match (sentence.id, sentence.begin_ms, sentence.end_ms) {
            (Some(id), Some(begin), Some(end)) => Some(format!("id:{id}:{begin}:{end}")),
            (Some(id), ..) => Some(format!("id:{id}")),
            (None, Some(begin), Some(end)) => Some(format!("{begin}:{end}")),
            _ => None,
        }
    }

    /// Applies an event.
    ///
    /// # Errors
    ///
    /// The task's failure, or `task-finished` before input was closed.
    pub(crate) fn on_event(&mut self, event: Event) -> Result<Vec<AsrEvent>, SpeechError> {
        let utterance = UtteranceId(self.utterance);
        match event {
            Event::Started => {
                if self.phase == Phase::Starting {
                    self.phase = Phase::Running;
                }
                Ok(Vec::new())
            }
            Event::Finished => {
                if self.phase != Phase::Finishing {
                    return Err(SpeechError::backend(
                        BACKEND,
                        true,
                        "the task finished before input closed",
                    ));
                }
                self.phase = Phase::Finished;
                Ok(Vec::new())
            }
            Event::Failed(error) => Err(error),
            Event::Result(sentence) if sentence.end => {
                if let Some(identity) = Self::identity(&sentence)
                    && !self.seen.insert(identity)
                {
                    return Ok(Vec::new());
                }
                self.utterance += 1;
                let mut out = Vec::new();
                self.start(sentence.begin_ms, &mut out);
                let start = sentence.begin_ms.map(millis).unwrap_or_default();
                let end = sentence
                    .end_ms
                    .map_or_else(|| self.sent_time(), millis)
                    .max(start);
                self.speech = None;
                self.last_stop = self.last_stop.max(end);
                out.push(AsrEvent::SpeechEnded { at: end, utterance });
                out.push(AsrEvent::Segment(Segment {
                    utterance,
                    text: sentence.text,
                    start,
                    end,
                }));
                self.report_activity(&mut out);
                Ok(out)
            }
            Event::Result(sentence) => {
                let mut out = Vec::new();
                self.start(sentence.begin_ms, &mut out);
                out.push(AsrEvent::Partial(Partial {
                    utterance,
                    text: sentence.text,
                }));
                Ok(out)
            }
            Event::Ignore => Ok(Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sentence(text: &str, end: bool, id: Option<f64>, times: Option<(f64, f64)>) -> Event {
        Event::Result(Sentence {
            text: text.into(),
            end,
            id,
            begin_ms: times.map(|t| t.0),
            end_ms: times.map(|t| t.1),
        })
    }

    /// The partials and segments among `events`.
    fn text(events: Vec<AsrEvent>) -> Vec<AsrEvent> {
        events
            .into_iter()
            .filter(|event| matches!(event, AsrEvent::Partial(_) | AsrEvent::Segment(_)))
            .collect()
    }

    #[test]
    fn partials_segments_and_dedup() {
        let mut state = DashScopeState::default();
        state.on_event(Event::Started).unwrap();
        assert_eq!(state.phase(), Phase::Running);
        let mut on = |event| text(state.on_event(event).unwrap());
        let events = on(sentence("你", false, Some(1.0), None));
        assert!(matches!(&events[..], [AsrEvent::Partial(p)] if p.utterance.0 == 0));
        let events = on(sentence("你好", true, Some(1.0), Some((0.0, 900.0))));
        let [AsrEvent::Segment(segment)] = &events[..] else {
            panic!("{events:?}")
        };
        assert_eq!(segment.end, Duration::from_millis(900));
        assert!(on(sentence("你好", true, Some(1.0), Some((0.0, 900.0)))).is_empty());
        // Same text, no identity: a legitimate repeat.
        assert_eq!(on(sentence("好", true, None, None)).len(), 1);
        assert_eq!(on(sentence("好", true, None, None)).len(), 1);
        assert_eq!(on(sentence("x", true, Some(9.0), None)).len(), 1);
        assert_eq!(on(sentence("y", true, None, Some((1.0, 2.0)))).len(), 1);
        assert!(on(Event::Ignore).is_empty());
    }

    #[test]
    fn sentences_are_the_speech() {
        let ms = Duration::from_millis;
        let mut state = DashScopeState::new(ms(1_000));
        state.on_event(Event::Started).unwrap();
        // 1.5 s sent and nothing heard: known to 0.5 s.
        assert_eq!(
            state.sent(24_000),
            [AsrEvent::ActivityKnown { through: ms(500) }]
        );
        let events = state
            .on_event(sentence("a", false, Some(1.0), Some((800.0, 0.0))))
            .unwrap();
        assert_eq!(events[0], AsrEvent::SpeechStarted { at: ms(800) });
        // While speech goes on, activity stays at its start.
        let events = state.sent(32_000);
        assert_eq!(events, [AsrEvent::ActivityKnown { through: ms(800) }]);
        assert!(state.sent(16_000).is_empty());
        let events = state
            .on_event(sentence("ab", true, Some(1.0), Some((800.0, 2_000.0))))
            .unwrap();
        assert_eq!(
            events[..2],
            [
                AsrEvent::SpeechEnded {
                    at: ms(2_000),
                    utterance: UtteranceId(0)
                },
                AsrEvent::Segment(Segment {
                    utterance: UtteranceId(0),
                    text: "ab".into(),
                    start: ms(800),
                    end: ms(2_000),
                }),
            ]
        );
        // 4.5 s sent: known to 3.5 s once speech ended.
        assert_eq!(events[2], AsrEvent::ActivityKnown { through: ms(3_500) });
        state.finishing();
        state.on_event(Event::Finished).unwrap();
        assert_eq!(
            state.finished(),
            [AsrEvent::ActivityKnown { through: ms(4_500) }]
        );
    }

    #[test]
    fn finishing() {
        let mut state = DashScopeState::default();
        assert!(state.on_event(Event::Finished).is_err());
        state.finishing();
        state.on_event(Event::Finished).unwrap();
        assert_eq!(state.phase(), Phase::Finished);
        let failed = Event::Failed(SpeechError::backend("x", false, "no"));
        assert!(DashScopeState::default().on_event(failed).is_err());
    }
}
