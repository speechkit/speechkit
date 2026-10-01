//! Ending turns and sessions at a point in the audio (A-07, A-08).
//!
//! The tracker sees every event a stream sends, with its times already
//! counting from the session's origin, and finds where turns end and where
//! the session should stop. Both need the backend to have confirmed the
//! silence: its `ActivityKnown` must have passed the end of speech plus the
//! silence, with no start of speech since, so a start reported late can't
//! be missed.

use std::{collections::VecDeque, time::Duration};

use crate::asr::{AsrEvent, AsrOptions, UtteranceId};

/// A turn whose silence was confirmed, published once its last utterance
/// is committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EndedTurn {
    /// Where its first speech started.
    pub(crate) start: Duration,
    /// Where its last speech ended.
    pub(crate) end: Duration,
    /// Its last utterance.
    pub(crate) last: UtteranceId,
}

/// What an event led to.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Observed {
    /// Turns to publish now, in order.
    pub(crate) turns: Vec<EndedTurn>,
    /// The session should stop taking audio: the endpoint was found.
    pub(crate) endpoint: bool,
}

/// Where turns end and where the session stops.
#[derive(Debug)]
pub(crate) struct Endpoints {
    turn_end: Option<Duration>,
    end_after_silence: Option<Duration>,
    /// The no-speech timeout, as a point from the origin.
    no_speech_until: Option<Duration>,
    /// Speech started since the session began.
    any_speech: bool,
    /// The last end of speech and its last utterance, while no speech has
    /// started since.
    last_end: Option<(Duration, UtteranceId)>,
    /// The last end of speech already confirmed as a turn end.
    turn_confirmed: bool,
    /// How far activity is known.
    known: Duration,
    /// The highest utterance committed.
    committed: Option<UtteranceId>,
    /// Where the current turn's first speech started.
    turn_start: Option<Duration>,
    /// Confirmed turns waiting for their text.
    pending: VecDeque<EndedTurn>,
    /// The endpoint was found.
    ended: bool,
}

impl Endpoints {
    /// Tracks the options of a session whose times start at `origin`.
    pub(crate) fn new(options: &AsrOptions, origin: Duration) -> Self {
        Self {
            turn_end: options.turn_end,
            end_after_silence: options.end_after_silence,
            no_speech_until: options
                .no_speech_timeout
                .map(|timeout| origin.saturating_add(timeout)),
            any_speech: false,
            last_end: None,
            turn_confirmed: false,
            known: origin,
            committed: None,
            turn_start: None,
            pending: VecDeque::new(),
            ended: false,
        }
    }

    /// Takes in `event`.
    pub(crate) fn observe(&mut self, event: &AsrEvent) -> Observed {
        match event {
            AsrEvent::SpeechStarted { at } => {
                self.any_speech = true;
                self.last_end = None;
                self.turn_confirmed = false;
                self.turn_start.get_or_insert(*at);
            }
            AsrEvent::SpeechEnded { at, utterance } => {
                self.last_end = Some((*at, *utterance));
                self.turn_confirmed = false;
            }
            AsrEvent::ActivityKnown { through } => self.known = self.known.max(*through),
            AsrEvent::Segment(segment) => {
                self.committed = self.committed.max(Some(segment.utterance));
            }
            AsrEvent::Partial(_) => {}
        }
        let mut observed = Observed::default();
        if let (Some(silence), Some((end, last))) = (self.turn_end, self.last_end)
            && !self.turn_confirmed
            && self.known >= end.saturating_add(silence)
        {
            self.turn_confirmed = true;
            let start = self.turn_start.take().unwrap_or(end);
            self.pending.push_back(EndedTurn { start, end, last });
        }
        while let Some(turn) = self.pending.front()
            && self.committed >= Some(turn.last)
        {
            observed.turns.extend(self.pending.pop_front());
        }
        if !self.ended {
            let pause = match (self.end_after_silence, self.last_end) {
                (Some(silence), Some((end, _))) => self.known >= end.saturating_add(silence),
                _ => false,
            };
            let no_speech = self
                .no_speech_until
                .is_some_and(|until| !self.any_speech && self.known >= until);
            if pause || no_speech {
                self.ended = true;
                observed.endpoint = true;
            }
        }
        observed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::Segment;

    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    fn started(at: u64) -> AsrEvent {
        AsrEvent::SpeechStarted { at: ms(at) }
    }

    fn ended(at: u64, utterance: u64) -> AsrEvent {
        AsrEvent::SpeechEnded {
            at: ms(at),
            utterance: UtteranceId(utterance),
        }
    }

    fn known(through: u64) -> AsrEvent {
        AsrEvent::ActivityKnown {
            through: ms(through),
        }
    }

    fn segment(utterance: u64) -> AsrEvent {
        AsrEvent::Segment(Segment {
            utterance: UtteranceId(utterance),
            text: String::new(),
            start: Duration::ZERO,
            end: Duration::ZERO,
        })
    }

    /// `Duration::MAX` means "never" for each option that takes a duration,
    /// and must not overflow the sums the endpointing does.
    #[test]
    fn the_longest_durations_mean_never() {
        let options = AsrOptions::default()
            .with_turn_end(Duration::MAX)
            .with_end_after_silence(Duration::MAX)
            .with_no_speech_timeout(Duration::MAX);
        // A listening does not start at zero.
        let mut endpoints = Endpoints::new(&options, Duration::from_secs(30));
        let seen = run(
            &mut endpoints,
            &[
                started(31_000),
                ended(32_000, 0),
                known(3_600_000),
                segment(0),
            ],
        );
        assert!(
            seen.iter().all(|o| o.turns.is_empty() && !o.endpoint),
            "{seen:?}"
        );
    }

    fn turns(options: &AsrOptions) -> Endpoints {
        Endpoints::new(options, Duration::ZERO)
    }

    fn run(endpoints: &mut Endpoints, events: &[AsrEvent]) -> Vec<Observed> {
        events
            .iter()
            .map(|event| endpoints.observe(event))
            .collect()
    }

    fn turn(start: u64, end: u64, last: u64) -> EndedTurn {
        EndedTurn {
            start: ms(start),
            end: ms(end),
            last: UtteranceId(last),
        }
    }

    #[test]
    fn a_turn_ends_once_its_silence_is_confirmed_and_its_text_is_in() {
        let mut endpoints = turns(&AsrOptions::default().with_turn_end(ms(700)));
        let seen = run(
            &mut endpoints,
            &[started(100), ended(1_000, 0), known(1_600)],
        );
        assert!(seen.iter().all(|o| o.turns.is_empty()));
        // Confirmed, but the segment is late.
        assert!(endpoints.observe(&known(1_700)).turns.is_empty());
        assert_eq!(endpoints.observe(&segment(0)).turns, [turn(100, 1_000, 0)]);
    }

    #[test]
    fn a_restart_reported_late_prevents_the_turn_end() {
        let options = AsrOptions::default()
            .with_turn_end(ms(700))
            .with_end_after_silence(ms(700));
        let mut endpoints = turns(&options);
        let seen = run(
            &mut endpoints,
            &[
                started(0),
                ended(1_000, 0),
                segment(0),
                known(1_500),
                // The restart at 1.6 s, reported after the input passed
                // 1.7 s, but before activity was known that far.
                started(1_600),
                known(2_000),
            ],
        );
        assert!(seen.iter().all(|o| o.turns.is_empty() && !o.endpoint));
        let seen = run(&mut endpoints, &[ended(2_500, 1), known(3_200), segment(1)]);
        assert!(seen[1].endpoint);
        // One turn, from the first start, once its text is in.
        assert_eq!(seen[2].turns, [turn(0, 2_500, 1)]);
    }

    #[test]
    fn a_turn_whose_text_is_late_keeps_only_its_own_utterances() {
        let mut endpoints = turns(&AsrOptions::default().with_turn_end(ms(700)));
        let seen = run(
            &mut endpoints,
            &[
                started(0),
                ended(1_000, 0),
                known(1_700),
                started(1_800),
                ended(2_000, 1),
                known(2_700),
            ],
        );
        assert!(seen.iter().all(|o| o.turns.is_empty()));
        // A's late segment releases A, then B.
        assert_eq!(endpoints.observe(&segment(0)).turns, [turn(0, 1_000, 0)]);
        assert_eq!(
            endpoints.observe(&segment(1)).turns,
            [turn(1_800, 2_000, 1)]
        );
    }

    #[test]
    fn a_cut_or_a_short_pause_never_ends_a_turn() {
        let mut endpoints = turns(&AsrOptions::default().with_turn_end(ms(700)));
        let seen = run(
            &mut endpoints,
            &[
                started(0),
                // A cut: a segment without an end of speech.
                segment(0),
                known(5_000),
                ended(6_000, 1),
                segment(1),
                known(6_500),
                started(6_600),
                known(7_000),
            ],
        );
        assert!(seen.iter().all(|o| o.turns.is_empty()));
    }

    #[test]
    fn no_speech_ends_the_session_at_its_timeout() {
        let options = AsrOptions::default().with_no_speech_timeout(ms(5_000));
        let mut endpoints = Endpoints::new(&options, ms(100_000));
        assert!(!endpoints.observe(&known(104_900)).endpoint);
        assert!(endpoints.observe(&known(105_000)).endpoint);
        assert!(!endpoints.observe(&known(106_000)).endpoint, "only once");
        let mut spoke = turns(&AsrOptions::default().with_no_speech_timeout(ms(1_000)));
        let seen = run(&mut spoke, &[started(500), known(2_000)]);
        assert!(seen.iter().all(|o| !o.endpoint));
    }
}
