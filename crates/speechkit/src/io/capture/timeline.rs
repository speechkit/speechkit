//! The one timeline of a capture, which every reader reads.
//!
//! Positions are frames at the device's rate, counted from the start of
//! the capture: capture time. The timeline holds the audio from the
//! earliest position anything still needs: the history, each reader's
//! cursor (less its look-back), and each reservation.
//!
//! Positions count the audio the device delivered. When it loses samples
//! (its buffer overflowed), the audio on either side of the gap is joined
//! in the timeline, so the timeline remembers where: a reader that would
//! read across the gap fails, and none starts before it.

use std::collections::{HashMap, VecDeque};

use crate::{
    Deadline, SpeechError,
    speech::{
        deadline::wait_until,
        sync::{Condvar, Mutex, MutexGuard, lock},
    },
};

/// What [`Timeline::read`] found.
#[derive(Debug, PartialEq)]
pub(crate) enum Read {
    /// The next samples, starting at the reader's cursor.
    Audio(Vec<f32>),
    /// Nothing new before the deadline.
    Idle,
    /// The reader reached its end, its stop or the capture's, at this
    /// position.
    End(u64),
    /// The reader lost audio at this position: it fell further behind than
    /// its limit, or the device lost samples before it got to them. The
    /// audio after it is gone.
    Lost(u64),
}

/// Where a reader stops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stop {
    /// It reads until the capture ends.
    Never,
    /// At the capture's position when the capture thread next runs, so it
    /// includes the audio the device has delivered but the thread has not
    /// moved yet.
    Requested,
    /// At this position.
    At(u64),
}

struct Reader {
    /// Where it started reading.
    start: u64,
    cursor: u64,
    /// Audio before the cursor that stays held, so a wake word heard in it
    /// can still be reserved.
    lookback: u64,
    /// How far behind the capture it may fall.
    limit: u64,
    stop: Stop,
    /// Whether it fell too far behind or would read across lost samples:
    /// it gets no more audio.
    lost: bool,
}

struct Reservation {
    at: u64,
    /// The capture position at which it lapses: `hold` after `at`.
    until: u64,
}

struct State {
    /// How far back a new reader may start, in frames.
    history: u64,
    /// How long a reservation holds, in frames.
    hold: u64,
    /// The position of `samples[0]`.
    start: u64,
    /// The earliest position whose audio is whole from there on: the
    /// device lost samples just before it.
    floor: u64,
    samples: VecDeque<f32>,
    ended: bool,
    readers: HashMap<u64, Reader>,
    reservations: HashMap<u64, Reservation>,
    next_id: u64,
}

impl State {
    fn end(&self) -> u64 {
        self.start + self.samples.len() as u64
    }

    fn id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// Where `reader` stops, once that is known.
    fn stop_of(&self, reader: &Reader) -> Option<u64> {
        match reader.stop {
            Stop::At(at) => Some(at),
            _ if self.ended => Some(self.end()),
            _ => None,
        }
    }

    /// Whether a read by `id` would return something other than `Idle`.
    fn ready(&self, id: u64) -> bool {
        let Some(reader) = self.readers.get(&id) else {
            return true;
        };
        reader.lost
            || reader.cursor < self.end()
            || self
                .stop_of(reader)
                .is_some_and(|stop| reader.cursor >= stop)
    }

    /// Resolves requested stops, marks readers that fell too far behind,
    /// lapses old reservations, and frees audio nothing needs.
    fn maintain(&mut self) {
        let end = self.end();
        for reader in self.readers.values_mut() {
            if reader.stop == Stop::Requested {
                reader.stop = Stop::At(end);
            }
            let upto = match reader.stop {
                Stop::At(at) => at.min(end),
                _ => end,
            };
            if !reader.lost && upto.saturating_sub(reader.cursor) > reader.limit {
                reader.lost = true;
            }
        }
        if self.ended {
            self.reservations.clear();
        } else {
            self.reservations
                .retain(|_, reservation| end < reservation.until);
        }
        self.trim();
    }

    fn trim(&mut self) {
        let end = self.end();
        let mut keep = if self.ended {
            end
        } else {
            end.saturating_sub(self.history)
        };
        for reader in self.readers.values().filter(|reader| !reader.lost) {
            keep = keep.min(reader.cursor.saturating_sub(reader.lookback));
        }
        for reservation in self.reservations.values() {
            keep = keep.min(reservation.at);
        }
        let drop = usize::try_from(keep.saturating_sub(self.start))
            .unwrap_or(usize::MAX)
            .min(self.samples.len());
        if drop > 0 {
            self.samples.drain(..drop);
            self.start += drop as u64;
            if self.samples.capacity() > 2 * self.samples.len().max(4_096) {
                self.samples.shrink_to_fit();
            }
        }
    }

    /// Records that the device lost samples at position `at`: the audio
    /// before it and the audio from it on are not contiguous. Every reader
    /// that started before it and does not stop at or before it would read
    /// across the gap, so it is lost (D-02); nothing can start before it
    /// (D-01), and a reservation before it lapses.
    fn lose(&mut self, at: u64) {
        // A stop requested before the loss ends where the timeline ends now.
        self.maintain();
        self.floor = self.floor.max(at);
        for reader in self.readers.values_mut() {
            let crosses = match reader.stop {
                Stop::At(stop) => stop > at,
                _ => true,
            };
            if reader.start < at && crosses {
                reader.lost = true;
            }
        }
        self.reservations
            .retain(|_, reservation| reservation.at >= at);
    }

    fn add_reader(&mut self, start: u64, lookback: u64, limit: u64) -> u64 {
        let id = self.id();
        self.readers.insert(
            id,
            Reader {
                start,
                cursor: start,
                lookback,
                limit,
                stop: Stop::Never,
                lost: false,
            },
        );
        id
    }
}

/// The audio of one capture, shared by the capture thread and every
/// reader.
pub(crate) struct Timeline {
    state: Mutex<State>,
    changed: Condvar,
}

impl Timeline {
    /// An empty timeline that keeps `history` frames for new readers and
    /// holds each reservation for `hold` frames.
    pub(crate) fn new(history: u64, hold: u64) -> Self {
        Self {
            state: Mutex::new(State {
                history,
                hold,
                start: 0,
                floor: 0,
                samples: VecDeque::new(),
                ended: false,
                readers: HashMap::new(),
                reservations: HashMap::new(),
                next_id: 0,
            }),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Appends what the device delivered since the last call, which may be
    /// nothing: every call also resolves stops and frees audio.
    pub(crate) fn append(&self, samples: &[f32]) {
        self.extend(samples, None);
    }

    /// [`append`](Self::append), and in the same step records that the
    /// device lost samples at position `lost_at` (see `State::lose`). The
    /// samples may include audio from after the loss. Readers see both
    /// together, so none reads across the gap first.
    pub(crate) fn append_with_loss(&self, samples: &[f32], lost_at: u64) {
        self.extend(samples, Some(lost_at));
    }

    fn extend(&self, samples: &[f32], lost_at: Option<u64>) {
        let mut state = self.lock();
        if state.ended {
            return;
        }
        state.samples.extend(samples);
        if let Some(at) = lost_at {
            state.lose(at);
        }
        state.maintain();
        drop(state);
        self.changed.notify_all();
    }

    /// Ends the timeline at its last sample: every reader ends there, and
    /// only what readers still need is kept.
    pub(crate) fn end(&self) {
        let mut state = self.lock();
        state.ended = true;
        state.maintain();
        drop(state);
        self.changed.notify_all();
    }

    /// The capture position: the audio captured so far.
    pub(crate) fn position(&self) -> u64 {
        self.lock().end()
    }

    /// Adds a reader at `start`, or at the current position if `None`,
    /// and returns its ID and start.
    ///
    /// # Errors
    ///
    /// [`SpeechError::Closed`] once the timeline has ended,
    /// [`SpeechError::Capacity`] if `start` is no longer held, or the
    /// device lost samples after it (D-01), or
    /// [`SpeechError::InvalidInput`] if it has not been captured yet.
    pub(crate) fn add_reader(
        &self,
        start: Option<u64>,
        lookback: u64,
        limit: u64,
    ) -> Result<(u64, u64), SpeechError> {
        let mut state = self.lock();
        if state.ended {
            return Err(SpeechError::Closed);
        }
        let end = state.end();
        let start = start.unwrap_or(end);
        if start > end {
            return Err(SpeechError::InvalidInput(
                "the listening starts after the audio captured so far".into(),
            ));
        }
        if start < state.start || start < state.floor {
            return Err(SpeechError::Capacity);
        }
        Ok((state.add_reader(start, lookback, limit), start))
    }

    /// Turns reservation `id` into a reader that starts at its position.
    ///
    /// # Errors
    ///
    /// [`SpeechError::Closed`] once the timeline has ended, or
    /// [`SpeechError::Capacity`] if the reservation lapsed (D-03).
    pub(crate) fn claim(&self, id: u64, limit: u64) -> Result<(u64, u64), SpeechError> {
        let mut state = self.lock();
        if state.ended {
            return Err(SpeechError::Closed);
        }
        let reservation = state
            .reservations
            .remove(&id)
            .ok_or(SpeechError::Capacity)?;
        let start = reservation.at;
        Ok((state.add_reader(start, 0, limit), start))
    }

    /// Reserves the audio from `at` on, if it is still held, and returns
    /// the reservation's ID. It lapses once the capture is `hold` past
    /// `at`.
    pub(crate) fn reserve(&self, at: u64) -> Option<u64> {
        let mut state = self.lock();
        if state.ended || at < state.start || at < state.floor || at > state.end() {
            return None;
        }
        let id = state.id();
        let until = at.saturating_add(state.hold);
        state.reservations.insert(id, Reservation { at, until });
        Some(id)
    }

    /// Releases reservation `id`, if it still exists.
    pub(crate) fn release(&self, id: u64) {
        let mut state = self.lock();
        if state.reservations.remove(&id).is_some() {
            state.trim();
        }
    }

    /// Reads up to `max` samples for reader `id`, waiting until
    /// `deadline` for audio or for its end.
    pub(crate) fn read(&self, id: u64, max: usize, deadline: Deadline) -> Read {
        let (mut state, _) = wait_until(&self.changed, self.lock(), deadline, |state| {
            state.ready(id)
        });
        let end = state.end();
        let start = state.start;
        let stop = state
            .readers
            .get(&id)
            .and_then(|reader| state.stop_of(reader));
        let Some(reader) = state.readers.get_mut(&id) else {
            return Read::End(end);
        };
        if reader.lost {
            return Read::Lost(reader.cursor);
        }
        let upto = stop.map_or(end, |stop| stop.min(end));
        if reader.cursor >= upto {
            return match stop {
                Some(_) => Read::End(reader.cursor),
                None => Read::Idle,
            };
        }
        let take = usize::try_from(upto - reader.cursor)
            .unwrap_or(usize::MAX)
            .min(max.max(1));
        let offset = usize::try_from(reader.cursor - start)
            .expect("a reader's cursor stays held, and the held audio fits in memory");
        let audio: Vec<f32> = state
            .samples
            .range(offset..offset + take)
            .copied()
            .collect();
        if let Some(reader) = state.readers.get_mut(&id) {
            reader.cursor += take as u64;
        }
        if state.ended {
            state.trim();
        }
        Read::Audio(audio)
    }

    /// Stops reader `id` at the capture's position as of the capture
    /// thread's next run.
    pub(crate) fn stop_reader(&self, id: u64) {
        let mut state = self.lock();
        let end = state.end();
        let ended = state.ended;
        if let Some(reader) = state.readers.get_mut(&id)
            && reader.stop == Stop::Never
        {
            reader.stop = if ended {
                Stop::At(end)
            } else {
                Stop::Requested
            };
        }
        drop(state);
        self.changed.notify_all();
    }

    /// Removes reader `id`, freeing the audio only it needed.
    pub(crate) fn remove_reader(&self, id: u64) {
        let mut state = self.lock();
        if state.readers.remove(&id).is_some() {
            state.trim();
        }
        drop(state);
        self.changed.notify_all();
    }

    /// Whether reader `id` lost audio: it fell further behind than its
    /// limit, or the device lost samples it had yet to read.
    pub(crate) fn is_lost(&self, id: u64) -> bool {
        self.lock()
            .readers
            .get(&id)
            .is_some_and(|reader| reader.lost)
    }

    /// Frames held, for tests of what is freed.
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        self.lock().samples.len()
    }
}
