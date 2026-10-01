//! The Realtime session state machine.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::Duration,
};

use crate::{
    SampleRate, SpeechError,
    asr::{AsrEvent, Partial, Segment, UtteranceId},
};

use super::protocol::{COMMIT_EMPTY, Event};

/// Where a Realtime session is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Waiting for the first `session.updated`.
    Connecting,
    /// Settings confirmed; no audio yet.
    Configured,
    /// Audio is flowing.
    Streaming,
    /// The final commit was sent; waiting for every item to complete.
    Committing,
    /// Done.
    Closed,
}

/// Where speech activity comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Activity {
    /// Nowhere: nothing is reported.
    None,
    /// The server's VAD, whose reports lag the audio sent by at most
    /// `lag`.
    Server {
        /// The stated maximum lag.
        lag: Duration,
    },
    /// The application's VAD, run by the stream.
    Client,
}

/// Turns server events into ordered backend events.
///
/// Items get utterance IDs in commit order: with server VAD, as soon as
/// their speech stops; with client VAD, when the stream commits them.
/// Deltas become partial results once their item has an ID, and
/// transcripts are released as segments strictly in that order, even when
/// the server completes items out of order. An item's audio ends where
/// server VAD says its speech stopped, or, without that, at the audio sent
/// when it was committed.
///
/// It also reports speech activity (see [`Activity`]): between speech,
/// activity is known up to the last end of speech, or the audio sent less
/// the lag, whichever is later.
#[derive(Debug)]
pub(crate) struct RealtimeState {
    phase: Phase,
    activity: Activity,
    /// The start of the speech in progress.
    speech: Option<Duration>,
    /// Where the last speech ended.
    last_stop: Duration,
    /// How far the client VAD knows activity.
    client_known: Duration,
    /// The last `ActivityKnown` sent.
    known: Duration,
    /// Items committed so far.
    committed: HashSet<String>,
    /// IDs the stream reserved for commits it sent, in order.
    reserved: VecDeque<u64>,
    /// The item behind each utterance ID.
    items: HashMap<String, u64>,
    next_index: u64,
    /// Accumulated delta text per item.
    texts: HashMap<String, String>,
    /// Items with deltas but no transcript yet.
    open: HashSet<String>,
    /// Items with a transcript.
    completed: HashSet<String>,
    /// Transcripts waiting for their turn, by utterance ID.
    ready: HashMap<u64, String>,
    /// Transcripts of items not yet committed.
    early: HashMap<String, String>,
    /// Where each committed, unreleased item's audio ends, by utterance ID.
    ends: HashMap<u64, Duration>,
    /// The next utterance ID to release.
    released: u64,
    commit_acked: bool,
    /// The commit sent while committing ends the stream's audio, not only
    /// this connection's, so it ends speech still in progress.
    final_commit: bool,
    /// Audio sent on this connection, at 24 kHz.
    frames: u64,
    /// The audio sent on earlier connections of the same stream, added to
    /// every time the server reports and to the audio sent.
    offset: Duration,
    last_end: Duration,
}

impl Default for RealtimeState {
    fn default() -> Self {
        Self::new(Activity::None)
    }
}

impl RealtimeState {
    /// A session that is connecting, reporting activity from `activity`.
    pub(crate) fn new(activity: Activity) -> Self {
        Self {
            phase: Phase::Connecting,
            activity,
            speech: None,
            last_stop: Duration::ZERO,
            client_known: Duration::ZERO,
            known: Duration::ZERO,
            committed: HashSet::new(),
            reserved: VecDeque::new(),
            items: HashMap::new(),
            next_index: 0,
            texts: HashMap::new(),
            open: HashSet::new(),
            completed: HashSet::new(),
            ready: HashMap::new(),
            early: HashMap::new(),
            ends: HashMap::new(),
            released: 0,
            commit_acked: false,
            final_commit: false,
            frames: 0,
            offset: Duration::ZERO,
            last_end: Duration::ZERO,
        }
    }

    /// The state for the next connection of the same stream, once this one
    /// has settled: utterance IDs, activity, and times go on from here.
    pub(crate) fn continued(&self) -> Self {
        let mut next = Self::new(self.activity);
        next.offset = self.sent_time();
        next.next_index = self.next_index;
        next.released = self.released;
        next.last_end = self.last_end;
        next.speech = self.speech;
        next.last_stop = self.last_stop;
        next.client_known = self.client_known;
        next.known = self.known;
        next
    }

    /// Whether speech is in progress, as far as the activity source says.
    pub(crate) fn speaking(&self) -> bool {
        self.speech.is_some()
    }

    /// The current phase.
    pub(crate) fn phase(&self) -> Phase {
        self.phase
    }

    /// Audio frames sent on this connection so far.
    pub(crate) fn frames(&self) -> u64 {
        self.frames
    }

    /// Records audio sent, and returns what that says about activity.
    pub(crate) fn sent(&mut self, frames: usize) -> Vec<AsrEvent> {
        self.frames += frames as u64;
        if self.phase == Phase::Configured {
            self.phase = Phase::Streaming;
        }
        let mut out = Vec::new();
        self.report_activity(&mut out);
        out
    }

    fn sent_time(&self) -> Duration {
        self.offset + SampleRate::HZ_24000.duration_of(self.frames)
    }

    /// Reserves the ID of the next item the stream commits.
    pub(crate) fn reserve(&mut self) -> UtteranceId {
        let id = self.fresh();
        self.reserved.push_back(id.0);
        id
    }

    /// An ID for an utterance that is never committed.
    pub(crate) fn fresh(&mut self) -> UtteranceId {
        let id = self.next_index;
        self.next_index += 1;
        UtteranceId(id)
    }

    /// The client VAD heard speech start at `at`.
    pub(crate) fn speech_started(&mut self, at: Duration) -> Vec<AsrEvent> {
        let mut out = Vec::new();
        if self.speech.is_none() {
            self.speech = Some(at);
            out.push(AsrEvent::SpeechStarted { at });
        }
        out
    }

    /// The client VAD heard speech end at `at`, with `utterance` last.
    pub(crate) fn speech_ended(&mut self, at: Duration, utterance: UtteranceId) -> Vec<AsrEvent> {
        let mut out = Vec::new();
        self.end_speech(at, utterance, &mut out);
        out
    }

    fn end_speech(&mut self, at: Duration, utterance: UtteranceId, out: &mut Vec<AsrEvent>) {
        if self.speech.take().is_some() {
            self.last_stop = self.last_stop.max(at);
            out.push(AsrEvent::SpeechEnded { at, utterance });
        }
    }

    /// Utterance `id`, never committed, has no words.
    pub(crate) fn dropped(&mut self, id: UtteranceId, end: Duration) -> Vec<AsrEvent> {
        let mut out = Vec::new();
        self.ready.insert(id.0, String::new());
        self.ends.insert(id.0, end);
        self.release(&mut out);
        out
    }

    /// The client VAD knows activity through `through`.
    pub(crate) fn client_known(&mut self, through: Duration) -> Vec<AsrEvent> {
        let mut out = Vec::new();
        self.client_known = self.client_known.max(through);
        self.report_activity(&mut out);
        out
    }

    /// The session settled: activity is known to the end of the audio.
    pub(crate) fn finished(&mut self) -> Vec<AsrEvent> {
        let mut out = Vec::new();
        if self.activity != Activity::None && self.speech.is_none() {
            let through = self.sent_time();
            if through > self.known {
                self.known = through;
                out.push(AsrEvent::ActivityKnown { through });
            }
        }
        out
    }

    /// Reports how far activity is known, if that moved.
    fn report_activity(&mut self, out: &mut Vec<AsrEvent>) {
        let candidate = match self.activity {
            Activity::None => return,
            Activity::Server { lag } => self.sent_time().saturating_sub(lag).max(self.last_stop),
            Activity::Client => self.client_known,
        };
        let through = self.speech.map_or(candidate, |start| candidate.min(start));
        if through > self.known {
            self.known = through;
            out.push(AsrEvent::ActivityKnown { through });
        }
    }

    /// The ID of `item_id`, giving it the next one if it has none.
    fn assign(&mut self, item_id: &str) -> u64 {
        if let Some(&index) = self.items.get(item_id) {
            return index;
        }
        let index = match self.reserved.pop_front() {
            Some(index) => index,
            None => self.fresh().0,
        };
        self.items.insert(item_id.to_owned(), index);
        index
    }

    /// The last commit of this connection is about to be sent (or nothing
    /// needs committing). It is the end of the stream when `last`; otherwise
    /// the stream goes on in the next connection, and speech in progress
    /// with it, so the commit does not end that speech.
    pub(crate) fn committing(&mut self, sent_commit: bool, last: bool) {
        self.phase = Phase::Committing;
        self.final_commit = last;
        if !sent_commit {
            self.commit_acked = true;
        }
    }

    /// Whether every committed item is released and nothing is pending.
    pub(crate) fn settled(&self) -> bool {
        self.phase == Phase::Committing
            && self.commit_acked
            && self.released == self.next_index
            && self.open.is_empty()
            && self.early.is_empty()
    }

    /// Marks the session closed.
    pub(crate) fn close(&mut self) {
        self.phase = Phase::Closed;
    }

    /// Applies a server event.
    ///
    /// # Errors
    ///
    /// The server's error, except an empty-commit error while committing.
    pub(crate) fn on_event(&mut self, event: Event) -> Result<Vec<AsrEvent>, SpeechError> {
        let mut out = Vec::new();
        match event {
            Event::SessionUpdated => {
                if self.phase == Phase::Connecting {
                    self.phase = Phase::Configured;
                }
            }
            Event::SpeechStarted { audio_start } => {
                let audio_start = audio_start + self.offset;
                if matches!(self.activity, Activity::Server { .. }) && self.speech.is_none() {
                    self.speech = Some(audio_start);
                    out.push(AsrEvent::SpeechStarted { at: audio_start });
                }
            }
            Event::SpeechStopped { item_id, audio_end } => {
                let audio_end = audio_end + self.offset;
                // Server VAD commits the item next, so it is the next
                // utterance.
                let index = self.assign(&item_id);
                if index >= self.released {
                    self.ends.entry(index).or_insert(audio_end);
                }
                if matches!(self.activity, Activity::Server { .. }) {
                    self.end_speech(audio_end, UtteranceId(index), &mut out);
                }
            }
            Event::Committed { item_id, .. } => {
                if self.phase == Phase::Committing {
                    self.commit_acked = true;
                }
                if self.committed.insert(item_id.clone()) {
                    let index = self.assign(&item_id);
                    let sent = self.sent_time();
                    let end = if index >= self.released {
                        *self.ends.entry(index).or_insert(sent)
                    } else {
                        sent
                    };
                    if self.phase == Phase::Committing
                        && self.final_commit
                        && matches!(self.activity, Activity::Server { .. })
                    {
                        // The final commit ends speech still in progress. A
                        // commit before a reconnect does not: the speech
                        // goes on in the next connection.
                        self.end_speech(end, UtteranceId(index), &mut out);
                    }
                    if let Some(text) = self.early.remove(&item_id) {
                        self.ready.insert(index, text);
                    } else if let Some(text) = self.texts.get(&item_id) {
                        out.push(Self::partial(index, text.clone()));
                    }
                }
            }
            Event::Delta { item_id, delta } => {
                if !self.completed.contains(&item_id) {
                    let text = self.texts.entry(item_id.clone()).or_default();
                    text.push_str(&delta);
                    let text = text.clone();
                    self.open.insert(item_id.clone());
                    if let Some(&index) = self.items.get(&item_id) {
                        out.push(Self::partial(index, text));
                    }
                }
            }
            Event::Completed {
                item_id,
                transcript,
            } => {
                if self.completed.insert(item_id.clone()) {
                    self.open.remove(&item_id);
                    self.texts.remove(&item_id);
                    match self.items.get(&item_id) {
                        Some(&index) => {
                            self.ready.insert(index, transcript);
                        }
                        None => {
                            self.early.insert(item_id, transcript);
                        }
                    }
                }
            }
            Event::Error { code, error } => {
                if self.phase == Phase::Committing && code.as_deref() == Some(COMMIT_EMPTY) {
                    self.commit_acked = true;
                } else {
                    return Err(error);
                }
            }
            Event::Other => {}
        }
        self.release(&mut out);
        self.report_activity(&mut out);
        Ok(out)
    }

    fn partial(index: u64, text: String) -> AsrEvent {
        AsrEvent::Partial(Partial {
            utterance: UtteranceId(index),
            text,
        })
    }

    fn release(&mut self, out: &mut Vec<AsrEvent>) {
        while let Some(text) = self.ready.remove(&self.released) {
            let end = self
                .ends
                .remove(&self.released)
                .unwrap_or_else(|| self.sent_time());
            out.push(AsrEvent::Segment(Segment {
                utterance: UtteranceId(self.released),
                text,
                start: self.last_end.min(end),
                end,
            }));
            self.last_end = end;
            self.released += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn committed(id: &str) -> Event {
        Event::Committed { item_id: id.into() }
    }

    fn delta(id: &str, text: &str) -> Event {
        Event::Delta {
            item_id: id.into(),
            delta: text.into(),
        }
    }

    fn completed(id: &str, text: &str) -> Event {
        Event::Completed {
            item_id: id.into(),
            transcript: text.into(),
        }
    }

    fn texts(events: &[AsrEvent]) -> Vec<String> {
        events
            .iter()
            .map(|event| match event {
                AsrEvent::Partial(p) => format!("p{}:{}", p.utterance.0, p.text),
                AsrEvent::Segment(s) => format!("s{}:{}", s.utterance.0, s.text),
                other => format!("{other:?}"),
            })
            .collect()
    }

    #[test]
    fn phases() {
        let mut state = RealtimeState::new(Activity::None);
        assert_eq!(state.phase(), Phase::Connecting);
        state.on_event(Event::SessionUpdated).unwrap();
        assert_eq!(state.phase(), Phase::Configured);
        state.sent(2_400);
        assert_eq!(state.phase(), Phase::Streaming);
        assert_eq!(state.frames(), 2_400);
        assert!(!state.settled());
        state.committing(false, true);
        assert!(state.settled());
        state.close();
        assert_eq!(state.phase(), Phase::Closed);
    }

    #[test]
    fn segments_follow_commit_order() {
        let mut state = RealtimeState::new(Activity::None);
        state.on_event(Event::SessionUpdated).unwrap();
        state.sent(24_000);
        assert_eq!(
            texts(&state.on_event(committed("a")).unwrap()),
            Vec::<String>::new()
        );
        assert_eq!(texts(&state.on_event(delta("a", "he")).unwrap()), ["p0:he"]);
        assert_eq!(
            texts(&state.on_event(delta("a", "llo")).unwrap()),
            ["p0:hello"]
        );
        state.on_event(committed("b")).unwrap();
        // b completes first but waits for a.
        assert!(state.on_event(completed("b", "world")).unwrap().is_empty());
        assert_eq!(
            texts(&state.on_event(completed("a", "hello")).unwrap()),
            ["s0:hello", "s1:world"]
        );
        // Duplicates and late deltas change nothing.
        assert!(state.on_event(completed("a", "again")).unwrap().is_empty());
        assert!(state.on_event(delta("a", "x")).unwrap().is_empty());
        state.committing(true, true);
        assert!(!state.settled());
        state.on_event(committed("c")).unwrap();
        assert!(!state.settled());
        assert_eq!(
            texts(&state.on_event(completed("c", "end")).unwrap()),
            ["s2:end"]
        );
        assert!(state.settled());
    }

    fn times(events: &[AsrEvent]) -> Vec<(Duration, Duration)> {
        events
            .iter()
            .filter_map(|event| match event {
                AsrEvent::Segment(s) => Some((s.start, s.end)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn segment_times_do_not_depend_on_when_transcripts_arrive() {
        let ms = Duration::from_millis;
        let mut state = RealtimeState::new(Activity::None);
        state.on_event(Event::SessionUpdated).unwrap();
        state.sent(36_000);
        state
            .on_event(Event::SpeechStopped {
                item_id: "a".into(),
                audio_end: ms(1_000),
            })
            .unwrap();
        state.on_event(committed("a")).unwrap();
        // The final commit, after all audio: ends at the audio sent.
        state.committing(true, true);
        state.on_event(committed("b")).unwrap();
        state.sent(24_000);
        assert_eq!(
            times(&state.on_event(completed("a", "one")).unwrap()),
            [(ms(0), ms(1_000))]
        );
        assert_eq!(
            times(&state.on_event(completed("b", "two")).unwrap()),
            [(ms(1_000), ms(1_500))]
        );
    }

    #[test]
    fn early_transcripts_wait_for_their_commit() {
        let mut state = RealtimeState::new(Activity::None);
        state.on_event(delta("x", "hi")).unwrap();
        assert!(state.on_event(completed("x", "hi.")).unwrap().is_empty());
        assert_eq!(texts(&state.on_event(committed("x")).unwrap()), ["s0:hi."]);
        state.on_event(delta("y", "yo")).unwrap();
        assert_eq!(texts(&state.on_event(committed("y")).unwrap()), ["p1:yo"]);
    }

    #[test]
    fn a_continued_state_goes_on_from_the_last() {
        let ms = Duration::from_millis;
        let mut first = RealtimeState::new(Activity::Server { lag: ms(1_000) });
        first.on_event(Event::SessionUpdated).unwrap();
        first.sent(48_000);
        first.on_event(committed("a")).unwrap();
        first.on_event(completed("a", "one")).unwrap();
        first.committing(false, false);
        assert!(first.settled());
        let mut next = first.continued();
        next.on_event(Event::SessionUpdated).unwrap();
        next.sent(24_000);
        let events = next
            .on_event(Event::SpeechStarted {
                audio_start: ms(200),
            })
            .unwrap();
        assert!(events.contains(&AsrEvent::SpeechStarted { at: ms(2_200) }));
        next.on_event(Event::SpeechStopped {
            item_id: "a".into(),
            audio_end: ms(800),
        })
        .unwrap();
        next.on_event(committed("a")).unwrap();
        let events = next.on_event(completed("a", "two")).unwrap();
        assert_eq!(texts(&events), ["s1:two"], "IDs go on across connections");
        assert_eq!(times(&events), [(ms(2_000), ms(2_800))]);
    }

    #[test]
    fn a_reconnect_does_not_end_speech_that_goes_on() {
        let ms = Duration::from_millis;
        let mut state = RealtimeState::new(Activity::Server { lag: ms(1_000) });
        state.on_event(Event::SessionUpdated).unwrap();
        state.sent(24_000);
        let started = state
            .on_event(Event::SpeechStarted {
                audio_start: ms(200),
            })
            .unwrap();
        assert!(started.contains(&AsrEvent::SpeechStarted { at: ms(200) }));
        // `wind_down(false)` while the user is speaking: reserve an ID, send
        // the commit, then the server acknowledges it.
        state.reserve();
        state.committing(true, false);
        let events = state.on_event(committed("a")).unwrap();
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AsrEvent::SpeechEnded { .. })),
            "the speech goes on in the next connection: {events:?}"
        );
        state.on_event(completed("a", "one")).unwrap();
        assert!(state.settled());
        // The next connection carries the speech on, ignores its own report
        // of a start, and ends it once, naming the next utterance.
        let mut next = state.continued();
        next.on_event(Event::SessionUpdated).unwrap();
        next.sent(12_000);
        let events = next
            .on_event(Event::SpeechStarted { audio_start: ms(0) })
            .unwrap();
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AsrEvent::SpeechStarted { .. })),
            "the speech had started already: {events:?}"
        );
        let events = next
            .on_event(Event::SpeechStopped {
                item_id: "b".into(),
                audio_end: ms(400),
            })
            .unwrap();
        assert!(events.contains(&AsrEvent::SpeechEnded {
            at: ms(1_400),
            utterance: UtteranceId(1),
        }));
        let events = next.on_event(committed("b")).unwrap();
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, AsrEvent::SpeechEnded { .. })),
            "the speech ends once: {events:?}"
        );
    }

    #[test]
    fn the_final_commit_ends_the_speech_in_progress() {
        let ms = Duration::from_millis;
        let mut state = RealtimeState::new(Activity::Server { lag: ms(1_000) });
        state.on_event(Event::SessionUpdated).unwrap();
        state.sent(24_000);
        state
            .on_event(Event::SpeechStarted {
                audio_start: ms(200),
            })
            .unwrap();
        state.reserve();
        state.committing(true, true);
        let events = state.on_event(committed("a")).unwrap();
        assert!(events.contains(&AsrEvent::SpeechEnded {
            at: ms(1_000),
            utterance: UtteranceId(0),
        }));
        assert!(!state.speaking());
    }

    #[test]
    fn empty_commit_error_only_while_committing() {
        let empty = || Event::Error {
            code: Some(COMMIT_EMPTY.into()),
            error: SpeechError::backend("x", false, "empty"),
        };
        let mut state = RealtimeState::new(Activity::None);
        assert!(state.on_event(empty()).is_err());
        state.committing(true, true);
        state.on_event(empty()).unwrap();
        assert!(state.settled());
    }
}
