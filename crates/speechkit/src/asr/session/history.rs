//! A session's history, and the readers that walk it.
//!
//! The history is one append-only log of segments and speech events, each
//! with a sequence number, plus the latest partial of each unfinished
//! utterance, tagged with the number at which it was published. A reader
//! keeps its own place, so there are no per-reader queues, and the session
//! never waits for a reader (A-05). A turn end stores the range of the log
//! that holds its segments, so its text is kept once.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use super::{Shared, endpoint::EndedTurn};
use crate::{
    Deadline, RecvError, SpeechError,
    asr::{AsrEvent, AsrResult, AsrUpdate, Partial, Segment, Turn, UtteranceId},
    speech::deadline,
};

/// The history cost of an entry besides its text, so events without text
/// still count against `max_history_bytes`.
const ENTRY_COST: usize = 64;

/// One logged item.
#[derive(Debug, Clone)]
enum Item {
    Segment(Segment),
    SpeechStarted(Duration),
    SpeechEnded(Duration, UtteranceId),
    /// A turn whose segments are the ones in `log[from..to]`.
    TurnEnded {
        from: usize,
        to: usize,
        start: Duration,
        end: Duration,
    },
}

#[derive(Debug)]
struct Entry {
    seq: u64,
    item: Item,
}

#[derive(Debug, Default)]
pub(crate) struct History {
    /// The last sequence number handed out.
    seq: u64,
    log: Vec<Entry>,
    /// The latest partial of each unfinished utterance, with its number.
    partials: BTreeMap<UtteranceId, (u64, String)>,
    /// Bytes counted against `max_history_bytes`.
    bytes: usize,
    /// Where the next turn's segments start in the log.
    turn_from: usize,
}

impl History {
    /// Records `event`, whose times already include the session's origin.
    ///
    /// # Errors
    ///
    /// [`SpeechError::Capacity`] if the history would pass `max_bytes`;
    /// nothing is recorded then.
    pub(crate) fn apply(&mut self, event: AsrEvent, max_bytes: usize) -> Result<(), SpeechError> {
        let item = match event {
            AsrEvent::Partial(Partial { utterance, text }) => {
                if self
                    .partials
                    .get(&utterance)
                    .is_some_and(|(_, old)| *old == text)
                {
                    return Ok(());
                }
                self.seq += 1;
                self.partials.insert(utterance, (self.seq, text));
                return Ok(());
            }
            AsrEvent::Segment(segment) => {
                self.partials.remove(&segment.utterance);
                // An utterance with no words ends its speech but reaches
                // neither readers nor the transcript.
                if segment.text.trim().is_empty() {
                    return Ok(());
                }
                Item::Segment(segment)
            }
            AsrEvent::SpeechStarted { at } => Item::SpeechStarted(at),
            AsrEvent::SpeechEnded { at, utterance } => Item::SpeechEnded(at, utterance),
            AsrEvent::ActivityKnown { .. } => return Ok(()),
        };
        self.log_item(item, max_bytes)
    }

    fn log_item(&mut self, item: Item, max_bytes: usize) -> Result<(), SpeechError> {
        let cost = ENTRY_COST
            + match &item {
                Item::Segment(segment) => segment.text.len(),
                Item::SpeechStarted(_) | Item::SpeechEnded(..) | Item::TurnEnded { .. } => 0,
            };
        if self.bytes + cost > max_bytes {
            return Err(SpeechError::Capacity);
        }
        self.bytes += cost;
        self.seq += 1;
        self.log.push(Entry {
            seq: self.seq,
            item,
        });
        Ok(())
    }

    /// Records the end of `turn`, whose segments are every one committed
    /// after the previous turn's, through `turn.last`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::Capacity`] as for [`apply`](Self::apply).
    pub(crate) fn turn_ended(
        &mut self,
        turn: EndedTurn,
        max_bytes: usize,
    ) -> Result<(), SpeechError> {
        let from = self.turn_from;
        let mut to = from;
        for (index, entry) in self.log.iter().enumerate().skip(from) {
            if let Item::Segment(segment) = &entry.item {
                if segment.utterance > turn.last {
                    break;
                }
                to = index + 1;
            }
        }
        self.log_item(
            Item::TurnEnded {
                from,
                to,
                start: turn.start,
                end: turn.end,
            },
            max_bytes,
        )?;
        self.turn_from = to;
        Ok(())
    }

    /// The committed segments, in order.
    pub(crate) fn segments(&self) -> Vec<Segment> {
        self.log
            .iter()
            .filter_map(|entry| match &entry.item {
                Item::Segment(segment) => Some(segment.clone()),
                _ => None,
            })
            .collect()
    }

    /// Drops the pending text, which a session that has ended abandons.
    pub(crate) fn abandon_partials(&mut self) {
        self.partials.clear();
    }

    /// A cursor for a reader that starts now.
    pub(crate) fn cursor(&self) -> Cursor {
        Cursor {
            log_pos: 0,
            joined: self.seq,
            caught_up: false,
            partial_seen: 0,
            closed: false,
        }
    }
}

/// Where one reader is in the history.
#[derive(Debug)]
pub(crate) struct Cursor {
    /// The next log entry to consider.
    log_pos: usize,
    /// The last number handed out when the reader joined.
    joined: u64,
    /// The segments from before the join have been delivered.
    caught_up: bool,
    /// Partials numbered up to this were delivered or replaced.
    partial_seen: u64,
    /// `Closed` was delivered.
    closed: bool,
}

impl Cursor {
    /// The next update, or [`RecvError::Empty`] if there is none yet.
    pub(crate) fn take(
        &mut self,
        history: &History,
        terminal: Option<&Arc<AsrResult>>,
    ) -> Result<AsrUpdate, RecvError> {
        if self.closed {
            return Err(RecvError::Closed);
        }
        if !self.caught_up {
            // Catch up on the segments. Speech and turn events from before
            // the join could only be stale, so they are skipped.
            while let Some(entry) = history.log.get(self.log_pos)
                && entry.seq <= self.joined
            {
                self.log_pos += 1;
                if let Item::Segment(segment) = &entry.item {
                    return Ok(AsrUpdate::Segment(segment.clone()));
                }
            }
            // Then the partials pending at the join, and everything newer.
            self.caught_up = true;
        }
        let logged = history.log.get(self.log_pos);
        let partial = history
            .partials
            .iter()
            .filter(|(_, (seq, _))| *seq > self.partial_seen)
            .min_by_key(|(_, (seq, _))| *seq);
        match (logged, partial) {
            (Some(entry), Some((_, (seq, _)))) if entry.seq < *seq => {
                Ok(self.logged(entry, history))
            }
            (Some(entry), None) => Ok(self.logged(entry, history)),
            (_, Some((&utterance, (seq, text)))) => {
                self.partial_seen = *seq;
                Ok(AsrUpdate::Partial(Partial {
                    utterance,
                    text: text.clone(),
                }))
            }
            (None, None) => match terminal {
                Some(result) => {
                    self.closed = true;
                    Ok(AsrUpdate::Closed(AsrResult::clone(result)))
                }
                None => Err(RecvError::Empty),
            },
        }
    }

    fn logged(&mut self, entry: &Entry, history: &History) -> AsrUpdate {
        self.log_pos += 1;
        match &entry.item {
            Item::Segment(segment) => AsrUpdate::Segment(segment.clone()),
            Item::SpeechStarted(at) => AsrUpdate::SpeechStarted { at: *at },
            Item::SpeechEnded(at, utterance) => AsrUpdate::SpeechEnded {
                at: *at,
                utterance: *utterance,
            },
            Item::TurnEnded {
                from,
                to,
                start,
                end,
            } => AsrUpdate::TurnEnded(Turn {
                segments: history.log[*from..*to]
                    .iter()
                    .filter_map(|entry| match &entry.item {
                        Item::Segment(segment) => Some(segment.clone()),
                        _ => None,
                    })
                    .collect(),
                start: *start,
                end: *end,
            }),
        }
    }
}

/// A reader of a session's updates: its partial text, segments, and
/// speech events, and finally [`AsrUpdate::Closed`] with the result.
///
/// Every reader receives every segment since the session started, and
/// every speech event since it started reading, exactly once and in order.
/// A slow reader skips partials that a newer one replaced, and never slows
/// the session (A-05). Dropping it never affects the session.
///
/// As an [`Iterator`], it waits for each update without a deadline and
/// ends after `Closed`; [`recv`](Self::recv) bounds the wait.
pub struct AsrUpdates {
    pub(super) shared: Arc<Shared>,
    pub(super) cursor: Cursor,
}

impl AsrUpdates {
    /// Waits for the next update until `deadline`. A timeout changes
    /// nothing.
    ///
    /// # Errors
    ///
    /// [`RecvError::Timeout`] if the deadline passed first, or
    /// [`RecvError::Closed`] after `Closed` was delivered.
    pub fn recv(&mut self, deadline: impl Into<Deadline>) -> Result<AsrUpdate, RecvError> {
        let deadline = deadline.into();
        let shared = &*self.shared;
        let cursor = &mut self.cursor;
        let mut taken = Err(RecvError::Empty);
        let _core = deadline::wait_until(&shared.changed, shared.lock(), deadline, |core| {
            taken = cursor.take(&core.history, shared.terminal.get());
            !matches!(taken, Err(RecvError::Empty))
        });
        taken.map_err(|error| match error {
            RecvError::Empty => RecvError::Timeout,
            other => other,
        })
    }

    /// The next update, without waiting.
    ///
    /// # Errors
    ///
    /// [`RecvError::Empty`] if none is ready, or [`RecvError::Closed`] after
    /// `Closed` was delivered.
    pub fn try_recv(&mut self) -> Result<AsrUpdate, RecvError> {
        let shared = &*self.shared;
        let core = shared.lock();
        self.cursor.take(&core.history, shared.terminal.get())
    }
}

impl Iterator for AsrUpdates {
    type Item = AsrUpdate;

    /// Waits for the next update. `None` after `Closed`.
    fn next(&mut self) -> Option<AsrUpdate> {
        let shared = &*self.shared;
        let cursor = &mut self.cursor;
        let mut taken = Err(RecvError::Empty);
        let _core = deadline::wait_forever(&shared.changed, shared.lock(), |core| {
            taken = cursor.take(&core.history, shared.terminal.get());
            !matches!(taken, Err(RecvError::Empty))
        });
        taken.ok()
    }
}

impl std::fmt::Debug for AsrUpdates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsrUpdates")
            .field("session", &self.shared.id)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[cfg(not(speechkit_loom))]
mod tests {
    use super::*;
    use crate::asr::Transcript;

    const MAX: usize = 1 << 20;

    fn segment(utterance: u64, text: &str) -> AsrEvent {
        AsrEvent::Segment(Segment {
            utterance: UtteranceId(utterance),
            text: text.into(),
            start: Duration::ZERO,
            end: Duration::ZERO,
        })
    }

    fn partial(utterance: u64, text: &str) -> AsrEvent {
        AsrEvent::Partial(Partial {
            utterance: UtteranceId(utterance),
            text: text.into(),
        })
    }

    fn drain(cursor: &mut Cursor, history: &History) -> Vec<String> {
        let mut seen = Vec::new();
        while let Ok(update) = cursor.take(history, None) {
            seen.push(match update {
                AsrUpdate::Partial(p) => format!("p{}:{}", p.utterance, p.text),
                AsrUpdate::Segment(s) => format!("s{}:{}", s.utterance, s.text),
                AsrUpdate::SpeechStarted { .. } => "started".into(),
                AsrUpdate::SpeechEnded { .. } => "ended".into(),
                AsrUpdate::TurnEnded(turn) => format!("turn:{}", turn.text()),
                AsrUpdate::Closed(_) => "closed".into(),
            });
        }
        seen
    }

    #[test]
    fn a_live_reader_sees_everything_in_order() {
        let mut history = History::default();
        let mut reader = history.cursor();
        history
            .apply(AsrEvent::SpeechStarted { at: Duration::ZERO }, MAX)
            .unwrap();
        history.apply(partial(1, "he"), MAX).unwrap();
        assert_eq!(drain(&mut reader, &history), ["started", "p1:he"]);
        history.apply(partial(1, "hel"), MAX).unwrap();
        history.apply(partial(1, "hello"), MAX).unwrap();
        history.apply(segment(1, "Hello."), MAX).unwrap();
        // The replaced partials were never read, so they are skipped.
        assert_eq!(drain(&mut reader, &history), ["s1:Hello."]);
    }

    #[test]
    fn a_late_reader_catches_up_on_segments_then_pending_partials() {
        let mut history = History::default();
        history
            .apply(AsrEvent::SpeechStarted { at: Duration::ZERO }, MAX)
            .unwrap();
        history.apply(partial(2, "pending"), MAX).unwrap();
        history.apply(segment(1, "one"), MAX).unwrap();
        let mut late = history.cursor();
        history.apply(segment(3, "three"), MAX).unwrap();
        assert_eq!(
            drain(&mut late, &history),
            ["s1:one", "p2:pending", "s3:three"]
        );
    }

    #[test]
    fn empty_segments_are_not_logged_and_the_bound_holds() {
        let mut history = History::default();
        history.apply(partial(1, "uh"), MAX).unwrap();
        history.apply(segment(1, "  "), MAX).unwrap();
        assert!(history.segments().is_empty());
        let mut reader = history.cursor();
        assert!(drain(&mut reader, &history).is_empty());
        let small = ENTRY_COST + 3;
        history.apply(segment(2, "abc"), small).unwrap();
        let full = history.apply(segment(3, "d"), small);
        assert!(matches!(full, Err(SpeechError::Capacity)));
        assert_eq!(history.segments().len(), 1);
    }

    #[test]
    fn a_turn_holds_its_own_segments() {
        let mut history = History::default();
        let mut reader = history.cursor();
        let turn = |last| EndedTurn {
            start: Duration::ZERO,
            end: Duration::ZERO,
            last: UtteranceId(last),
        };
        history.apply(segment(0, "a"), MAX).unwrap();
        history.apply(segment(1, " "), MAX).unwrap();
        history.apply(segment(2, "b"), MAX).unwrap();
        history.apply(segment(3, "c"), MAX).unwrap();
        history.turn_ended(turn(2), MAX).unwrap();
        history.turn_ended(turn(3), MAX).unwrap();
        // A cough: no segments.
        history.turn_ended(turn(4), MAX).unwrap();
        assert_eq!(
            drain(&mut reader, &history),
            ["s0:a", "s2:b", "s3:c", "turn:a b", "turn:c", "turn:"]
        );
        // A late reader catches up on segments, not turns.
        let mut late = history.cursor();
        assert_eq!(drain(&mut late, &history), ["s0:a", "s2:b", "s3:c"]);
    }

    #[test]
    fn closed_is_last() {
        let mut history = History::default();
        history.apply(segment(1, "one"), MAX).unwrap();
        let mut reader = history.cursor();
        let result = Arc::new(Ok(Transcript::default()));
        assert!(matches!(
            reader.take(&history, Some(&result)),
            Ok(AsrUpdate::Segment(_))
        ));
        assert!(matches!(
            reader.take(&history, Some(&result)),
            Ok(AsrUpdate::Closed(Ok(_)))
        ));
        assert_eq!(
            reader.take(&history, Some(&result)).err(),
            Some(RecvError::Closed)
        );
    }
}

#[cfg(test)]
#[cfg(speechkit_loom)]
mod loom_tests {
    use loom::sync::{Arc, Mutex};

    use super::*;

    /// Two readers and a writer: each reader sees every segment once, in
    /// order, however the threads interleave. Each reader reads twice while
    /// the writer runs, then drains what is left once everyone is done.
    #[test]
    fn history_readers_see_every_segment_once_in_order() {
        loom::model(|| {
            let history = Arc::new(Mutex::new(History::default()));
            let readers: Vec<_> = (0..2)
                .map(|_| {
                    let history = history.clone();
                    loom::thread::spawn(move || {
                        let mut cursor = history.lock().unwrap().cursor();
                        let mut seen = Vec::new();
                        for _ in 0..2 {
                            let taken = cursor.take(&history.lock().unwrap(), None);
                            if let Ok(AsrUpdate::Segment(segment)) = taken {
                                seen.push(segment.utterance.0);
                            }
                        }
                        (cursor, seen)
                    })
                })
                .collect();
            for utterance in 1..=2 {
                let segment = AsrEvent::Segment(Segment {
                    utterance: UtteranceId(utterance),
                    text: "x".into(),
                    start: Duration::ZERO,
                    end: Duration::ZERO,
                });
                history.lock().unwrap().apply(segment, 1 << 20).unwrap();
            }
            for reader in readers {
                let (mut cursor, mut seen) = reader.join().unwrap();
                let locked = history.lock().unwrap();
                while let Ok(AsrUpdate::Segment(segment)) = cursor.take(&locked, None) {
                    seen.push(segment.utterance.0);
                }
                assert_eq!(seen, [1, 2]);
            }
        });
    }
}
