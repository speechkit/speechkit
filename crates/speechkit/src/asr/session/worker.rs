//! The session thread: waits for a slot, opens the stream, and feeds it.

use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    time::Duration,
};

use super::{AsrEvents, Shared};
use crate::{
    SampleRate, SpeechError,
    asr::{AsrBackend, AsrOptions, AsrStream},
    speech::{deadline, opening::Stage, resample::Resampler, slots::SlotGuard},
};

/// Everything a session's thread owns before its stream opens.
pub(crate) struct Worker {
    pub(crate) shared: Arc<Shared>,
    pub(crate) backend: Arc<dyn AsrBackend>,
    pub(crate) options: AsrOptions,
    /// The rate of the audio pushed.
    pub(crate) from: SampleRate,
    /// The backend's rate. The thread builds the resampler between the two
    /// once it holds a slot, so `start` does no CPU-heavy work.
    pub(crate) to: SampleRate,
    /// Frames fed to the backend at a time, at its rate.
    pub(crate) block: usize,
    /// The session ends after this much audio.
    pub(crate) max_length: Option<Duration>,
    /// A slot the caller already took, or `None` to wait for one.
    pub(crate) slot: Option<SlotGuard>,
}

impl Worker {
    /// Runs the session to its end on the current thread.
    pub(crate) fn run(self) {
        let Self {
            shared,
            backend,
            options,
            from,
            to,
            block,
            max_length,
            slot,
        } = self;
        let ended = || shared.terminal.get().is_some();
        let slot = match slot {
            Some(slot) => slot,
            None => match shared.slots.acquire(ended) {
                Some(slot) => slot,
                None => return,
            },
        };
        {
            let mut core = shared.lock();
            if ended() {
                return;
            }
            core.stage = Stage::Backend;
        }
        let name = backend.name().to_owned();
        let events = AsrEvents::session(shared.clone());
        // A failure to build the resampler ends the session like a failure to
        // open the stream.
        let opened = Resampler::new(from, to).and_then(|resampler| {
            if ended() {
                return Err(SpeechError::Closed);
            }
            let stream = catch_unwind(AssertUnwindSafe(|| backend.open(&options, events)))
                .unwrap_or_else(|_| {
                    Err(SpeechError::backend(
                        name.clone(),
                        false,
                        "the backend panicked while opening a stream",
                    ))
                })?;
            Ok((resampler, stream))
        });
        let (resampler, stream) = match opened {
            Ok(opened) => opened,
            Err(error) => {
                // The slot goes before the result is published (A-09).
                drop(slot);
                let mut core = shared.lock();
                shared.settle_failure(&mut core, error);
                return;
            }
        };
        {
            let mut core = shared.lock();
            if !ended() {
                core.stage = Stage::Open;
                tracing::debug!(session = shared.id, backend = %name, "session started");
            }
        }
        shared.changed.notify_all();
        let rate = resampler.to_rate();
        Running {
            shared,
            backend: name,
            stream,
            resampler,
            block,
            pending: Vec::new(),
            fed: 0,
            rate,
            max_length,
            slot,
        }
        .run();
    }
}

/// A session with an open stream.
struct Running {
    shared: Arc<Shared>,
    backend: String,
    stream: Box<dyn AsrStream>,
    resampler: Resampler,
    block: usize,
    /// Resampled audio waiting for a full block.
    pending: Vec<f32>,
    /// Frames fed to the backend, at its rate.
    fed: u64,
    /// The backend's rate.
    rate: SampleRate,
    /// The session ends after this much audio.
    max_length: Option<Duration>,
    /// Released only after the stream is dropped, so work it still runs
    /// counts against the limit, but before the result is published
    /// (A-09).
    slot: SlotGuard,
}

/// What the session thread does next.
enum Next {
    Chunk(Vec<f32>),
    /// The input ended: feed what is left, then finish.
    Drain,
    /// The session reached its cutoff: finish at once.
    Cut,
    Stop,
}

/// Whether feeding goes on.
#[derive(PartialEq, Eq)]
enum Fed {
    More,
    /// The session reached its cutoff.
    Cut,
    Stop,
}

impl Running {
    fn run(mut self) {
        let outcome = catch_unwind(AssertUnwindSafe(|| self.drive()));
        let mut error = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error),
            Err(_) => Some(self.panicked()),
        };
        let Self {
            shared,
            mut stream,
            slot,
            ..
        } = self;
        {
            let mut core = shared.lock();
            if let Some(failure) = core.failure.take() {
                error = Some(failure);
            }
        }
        // Stop a stream that failed or was ended from outside. Its late
        // events are discarded (A-03).
        if error.is_some() || shared.terminal.get().is_some() {
            let _ = catch_unwind(AssertUnwindSafe(|| stream.cancel()));
        }
        // Every backend call has returned, so the stream and the slot go
        // before the result is published: a caller woken by `finish` can
        // start the next session at once (A-09). If the session already
        // ended from outside, the slot was held until now.
        let _ = catch_unwind(AssertUnwindSafe(|| drop(stream)));
        drop(slot);
        let mut core = shared.lock();
        match error {
            Some(error) => shared.settle_failure(&mut core, error),
            None if shared.terminal.get().is_none() => {
                let transcript = shared.transcript(&core);
                shared.settle(&mut core, Ok(transcript));
            }
            None => {}
        }
    }

    fn drive(&mut self) -> Result<(), SpeechError> {
        loop {
            match self.next() {
                Next::Chunk(chunk) => {
                    self.resampler.process(&chunk, &mut self.pending)?;
                    while self.pending.len() >= self.block {
                        let rest = self.pending.split_off(self.block);
                        let block = std::mem::replace(&mut self.pending, rest);
                        match self.feed(block)? {
                            Fed::More => {}
                            Fed::Cut => return self.call(|stream| stream.finish()),
                            Fed::Stop => return Ok(()),
                        }
                    }
                }
                Next::Drain => break,
                Next::Cut => return self.call(|stream| stream.finish()),
                Next::Stop => return Ok(()),
            }
        }
        // A partial block is fed only at the end of the input.
        self.resampler.flush(&mut self.pending)?;
        if !self.pending.is_empty() {
            let tail = std::mem::take(&mut self.pending);
            if self.feed(tail)? == Fed::Stop {
                return Ok(());
            }
        }
        self.call(|stream| stream.finish())
    }

    /// Feeds `block`, cut exactly at the maximum length (A-07), and says
    /// whether to go on. The events it confirms may set the cutoff.
    fn feed(&mut self, mut block: Vec<f32>) -> Result<Fed, SpeechError> {
        let shared = self.shared.clone();
        let limit = self.max_length.map(|length| self.rate.frames_in(length));
        let at_limit = limit.is_some_and(|limit| {
            let room = usize::try_from(limit.saturating_sub(self.fed)).unwrap_or(usize::MAX);
            block.truncate(room);
            block.len() == room
        });
        if !block.is_empty() {
            self.fed += block.len() as u64;
            let position = shared.origin + self.rate.duration_of(self.fed);
            shared.lock().position = position;
            self.call(|stream| stream.accept(&block))?;
        }
        let mut core = shared.lock();
        if shared.stopping(&core) {
            return Ok(Fed::Stop);
        }
        if at_limit {
            let at = shared.origin + self.rate.duration_of(self.fed);
            shared.cut(&mut core, at);
        }
        Ok(if core.cutoff.is_some() {
            Fed::Cut
        } else {
            Fed::More
        })
    }

    /// Waits for the next chunk, the end of input, or the end of the
    /// session.
    fn next(&self) -> Next {
        let shared = &*self.shared;
        let mut core = deadline::wait_forever(&shared.changed, shared.lock(), |core| {
            shared.stopping(core) || !core.queue.is_empty() || core.input_closed
        });
        if shared.stopping(&core) {
            return Next::Stop;
        }
        if core.cutoff.is_some() {
            return Next::Cut;
        }
        match core.queue.pop_front() {
            Some(chunk) => {
                core.queued -= chunk.len();
                core.consumed += chunk.len() as u64;
                drop(core);
                shared.changed.notify_all();
                Next::Chunk(chunk)
            }
            None => Next::Drain,
        }
    }

    /// Calls the stream, turning a panic into a backend error (A-09).
    fn call(
        &mut self,
        f: impl FnOnce(&mut dyn AsrStream) -> Result<(), SpeechError>,
    ) -> Result<(), SpeechError> {
        let stream = &mut *self.stream;
        catch_unwind(AssertUnwindSafe(|| f(stream))).unwrap_or_else(|_| Err(self.panicked()))
    }

    fn panicked(&self) -> SpeechError {
        SpeechError::backend(self.backend.clone(), false, "the backend panicked")
    }
}
