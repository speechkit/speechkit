//! Speaker playback with cpal: one output stream and a queue of sounds.
//!
//! The player thread moves each sound, resampled to the device's rate,
//! into a lock-free ring buffer, one sound after another. Every sample it
//! writes gets the next index; a playback is a range of indices. The
//! output callback reads the ring, skips ranges that were stopped, and
//! publishes the index it reached with the time its first sample will be
//! played, which is how `played()` counts only audio already heard
//! (D-06).

use std::{
    collections::VecDeque,
    sync::{
        Arc, Condvar, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering, fence},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use cpal::traits::{DeviceTrait, StreamTrait};
use ringbuf::{
    HeapCons, HeapProd, HeapRb,
    traits::{Consumer, Producer, Split},
};

use crate::{
    Deadline, RecvError, SampleRate, SpeechError,
    io::{
        DeviceInfo, Opened,
        convert::{OutputSample, fill_interleaved},
        list, open,
    },
    speech::{deadline::wait_until, opening::Opening, resample::Resampler},
    tts::{PushedText, TtsEngine, TtsOptions, TtsSession, TtsUpdate},
};

mod fake;
mod sound;

pub use fake::FakeSpeaker;
use sound::{Channel, Source, Taken};
pub use sound::{Sink, Sound};

/// The ring buffer holds this much audio: what `stop` has to skip.
const BUFFER: Duration = Duration::from_millis(500);
/// How often the player thread checks the device and the queue.
const TICK: Duration = Duration::from_millis(5);
/// The most audio taken from a sound at once.
const PIECE: Duration = Duration::from_millis(100);
/// Silence before this much of a sound has been queued is not an underrun.
const PREFILL: Duration = Duration::from_millis(100);
/// The most ranges the callback skips at once; more wait their turn.
const SKIPS: usize = 4;
/// The skip ranges the player publishes: `from, to` pairs.
type Skips = [u64; 2 * SKIPS];
/// Added to playback times, in nanoseconds, so a time before the clock
/// started still counts forward.
const BIAS: u64 = 1 << 62;

fn device_error(error: impl std::fmt::Display) -> SpeechError {
    SpeechError::backend("speaker", true, error.to_string())
}

fn lost_error() -> SpeechError {
    device_error("the output device stopped")
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `N` values one thread writes and others read consistently, without
/// locking the writer: a sequence lock.
struct SeqCell<const N: usize> {
    seq: AtomicU64,
    values: [AtomicU64; N],
}

impl<const N: usize> SeqCell<N> {
    fn new() -> Self {
        Self {
            seq: AtomicU64::new(0),
            values: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Only one thread may write.
    fn write(&self, values: [u64; N]) {
        let seq = self.seq.load(Ordering::Relaxed);
        self.seq.store(seq.wrapping_add(1), Ordering::Relaxed);
        fence(Ordering::Release);
        for (cell, value) in self.values.iter().zip(values) {
            cell.store(value, Ordering::Relaxed);
        }
        self.seq.store(seq.wrapping_add(2), Ordering::Release);
    }

    /// One attempt at reading: `None` while a write is in progress. For a
    /// reader that must not wait for the writer, such as the output
    /// callback.
    fn try_read(&self) -> Option<[u64; N]> {
        let before = self.seq.load(Ordering::Acquire);
        if before % 2 == 1 {
            return None;
        }
        let values = std::array::from_fn(|index| self.values[index].load(Ordering::Relaxed));
        fence(Ordering::Acquire);
        (self.seq.load(Ordering::Relaxed) == before).then_some(values)
    }

    /// Reads, spinning while the writer is busy. Only for threads that may
    /// wait for the writer.
    fn read(&self) -> [u64; N] {
        loop {
            if let Some(values) = self.try_read() {
                return values;
            }
            std::hint::spin_loop();
        }
    }
}

/// State shared by the speaker, its playbacks, the player thread, and the
/// output callback.
pub(crate) struct Shared {
    rate: SampleRate,
    /// When the clock started; playback times are nanoseconds since it.
    base: Instant,
    /// From the callback: the index its latest buffer started at, when
    /// that buffer starts playing, and how many samples it had.
    clock: SeqCell<3>,
    /// From the player thread: up to [`SKIPS`] ranges of indices the
    /// callback skips, as `from, to` pairs in index order; empty ones are
    /// `0, 0`.
    skip: SeqCell<{ 2 * SKIPS }>,
    /// The index the callback has reached, skipped samples included.
    popped: AtomicU64,
    /// A sound is playing, so silence is an underrun.
    expecting: AtomicBool,
    underruns: AtomicU64,
    lost: AtomicBool,
    closing: AtomicBool,
    queue: Mutex<Queue>,
    changed: Condvar,
    /// How many times the player's loop has run, for the test that it does
    /// not spin.
    #[cfg(test)]
    loops: AtomicU64,
}

#[derive(Default)]
struct Queue {
    waiting: VecDeque<(Arc<PlaybackState>, Sound)>,
    /// `Speaker::stop` was called: the player stops what it plays.
    stop_all: bool,
}

impl Shared {
    fn new(rate: SampleRate) -> Self {
        Self {
            rate,
            base: Instant::now(),
            clock: SeqCell::new(),
            skip: SeqCell::new(),
            popped: AtomicU64::new(0),
            expecting: AtomicBool::new(false),
            underruns: AtomicU64::new(0),
            lost: AtomicBool::new(false),
            closing: AtomicBool::new(false),
            queue: Mutex::new(Queue::default()),
            changed: Condvar::new(),
            #[cfg(test)]
            loops: AtomicU64::new(0),
        }
    }

    /// Nanoseconds since the clock started, plus [`BIAS`].
    fn now(&self) -> u64 {
        u64::try_from(self.base.elapsed().as_nanos())
            .unwrap_or(u64::MAX)
            .saturating_add(BIAS)
    }

    /// The index up to which the device has played: samples before it
    /// have reached their playback time. Samples skipped within the
    /// latest buffer count as if played, so this is never ahead of a
    /// sound played after them.
    fn played_index(&self) -> u64 {
        let [start, at, count] = self.clock.read();
        let elapsed = Duration::from_nanos(self.now().saturating_sub(at));
        start + self.rate.frames_in(elapsed).min(count)
    }

    /// Fills `out` from the ring, as the output callback does. Its first
    /// frame plays `latency` from now, or played `ago`.
    ///
    /// The callback runs in real time, so it never waits for the player:
    /// `skips` holds the ranges it last saw, and is refreshed only if the
    /// player is not publishing new ones right now. At worst a few
    /// milliseconds of a stopped sound play once more.
    fn render<T: OutputSample>(
        &self,
        consumer: &mut impl Consumer<Item = f32>,
        out: &mut [T],
        channels: usize,
        latency: Duration,
        ago: Duration,
        skips: &mut Skips,
    ) {
        if let Some(fresh) = self.skip.try_read() {
            *skips = fresh;
        }
        let skips = *skips;
        let mut index = self.popped.load(Ordering::Relaxed);
        let skip = |consumer: &mut dyn FnMut(usize) -> usize, index: &mut u64| {
            // Ranges are in index order, so one pass skips them all.
            for &[from, to] in skips.as_chunks::<2>().0 {
                if (from..to).contains(index) {
                    let wanted = usize::try_from(to - *index).unwrap_or(usize::MAX);
                    *index += consumer(wanted) as u64;
                }
            }
        };
        skip(&mut |count| consumer.skip(count), &mut index);
        let start = index;
        let mut real = 0;
        let mut dry = false;
        let silent = fill_interleaved(out, channels, || {
            if dry {
                return None;
            }
            let sample = consumer.try_pop();
            match sample {
                Some(_) => {
                    index += 1;
                    real += 1;
                    skip(&mut |count| consumer.skip(count), &mut index);
                }
                None => dry = true,
            }
            sample
        });
        self.popped.store(index, Ordering::Relaxed);
        let at = self
            .now()
            .saturating_add(u64::try_from(latency.as_nanos()).unwrap_or(0));
        let at = at.saturating_sub(u64::try_from(ago.as_nanos()).unwrap_or(u64::MAX));
        self.clock.write([start, at, real]);
        if silent > 0 && self.expecting.load(Ordering::Relaxed) {
            self.underruns.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Fills mono `out` as a callback whose audio has all been heard, for
    /// the fake speaker.
    fn render_played(
        &self,
        consumer: &mut impl Consumer<Item = f32>,
        out: &mut [f32],
        took: Duration,
        skips: &mut Skips,
    ) {
        self.render(consumer, out, 1, Duration::ZERO, took, skips);
    }

    fn device_lost(&self) -> bool {
        self.lost.load(Ordering::Relaxed)
    }
}

/// Where a speaker's audio goes. The device's payload is boxed because its
/// size depends on the platform's audio backend, and `Fake` carries nothing.
enum Output {
    Device(Box<DeviceOutput>),
    Fake,
}

/// A real output device, and the stream that plays through it.
struct DeviceOutput {
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    /// The ring's reading end, until the stream starts.
    consumer: Mutex<Option<HeapCons<f32>>>,
    stream: Mutex<Option<cpal::Stream>>,
}

/// An output device, and the queue of sounds it plays.
///
/// Sounds play one after another, in the order they were queued, each
/// resampled to the device's rate: a synthesis
/// ([`speak`](Self::speak), [`speak_streaming`](Self::speak_streaming),
/// or [`play`](Self::play) with a [`TtsOutput`](crate::tts::TtsOutput)),
/// an [`AudioBuffer`](crate::AudioBuffer), or samples pushed into a [`sink`](Self::sink). Each
/// returns a [`Playback`] to follow, stop, or wait for it. The output
/// stream starts with the first sound and runs until the speaker is
/// dropped.
///
/// Silence while audio was due is counted by
/// [`underruns`](Self::underruns) and logged at `warn`. A lost device
/// fails every queued playback (D-05).
pub struct Speaker {
    shared: Arc<Shared>,
    output: Output,
    name: String,
    channels: usize,
    player: Option<JoinHandle<()>>,
}

impl Speaker {
    /// The system's default output device.
    ///
    /// # Errors
    ///
    /// A retryable backend error if there is no output device or it has
    /// no usable configuration, or `Unsupported` for its sample format.
    pub fn open_default() -> Result<Self, SpeechError> {
        Self::from_device(open(false, None)?)
    }

    /// The output device called `name`, as [`list`](Self::list) lists it.
    /// The exact name wins, then the name ignoring case, then a part of
    /// one name ignoring case.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] if `name` is empty, matches no output
    /// device, or matches several, including devices with the same name;
    /// a retryable backend error if the devices cannot be listed or the
    /// device has no usable configuration; or `Unsupported` for its sample
    /// format.
    pub fn open(name: &str) -> Result<Self, SpeechError> {
        Self::from_device(open(false, Some(name))?)
    }

    /// The output devices, in the order the system lists them.
    ///
    /// # Errors
    ///
    /// A retryable backend error if the devices cannot be listed.
    pub fn list() -> Result<Vec<DeviceInfo>, SpeechError> {
        list(false)
    }

    /// A speaker with no device, and the handle a test plays it through.
    /// For the device contract tests; not covered by semver.
    #[doc(hidden)]
    pub fn fake(sample_rate: SampleRate) -> (Self, FakeSpeaker) {
        let (producer, consumer) = ring(sample_rate);
        let shared = Arc::new(Shared::new(sample_rate));
        let speaker = Self::start(shared.clone(), producer, Output::Fake, "fake".into(), 1);
        let speaker = speaker.expect("a thread for the fake speaker");
        (speaker, FakeSpeaker::new(shared, consumer))
    }

    fn from_device(opened: Opened) -> Result<Self, SpeechError> {
        let (producer, consumer) = ring(opened.rate);
        let channels = usize::from(opened.config.channels());
        let output = Output::Device(Box::new(DeviceOutput {
            device: opened.device,
            config: opened.config,
            consumer: Mutex::new(Some(consumer)),
            stream: Mutex::new(None),
        }));
        let shared = Arc::new(Shared::new(opened.rate));
        Self::start(shared, producer, output, opened.name, channels)
    }

    fn start(
        shared: Arc<Shared>,
        producer: HeapProd<f32>,
        output: Output,
        name: String,
        channels: usize,
    ) -> Result<Self, SpeechError> {
        let player = Player::new(shared.clone(), producer);
        let player = std::thread::Builder::new()
            .name("speechkit-speaker".into())
            .spawn(move || player.run())
            .map_err(device_error)?;
        Ok(Self {
            shared,
            output,
            name,
            channels,
            player: Some(player),
        })
    }

    /// The device's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The device's rate; every sound is resampled to it.
    pub fn sample_rate(&self) -> SampleRate {
        self.shared.rate
    }

    /// Synthesizes `text` on `engine` and plays it as the audio arrives.
    /// It checks the options at once and returns; the synthesis opens in
    /// the background.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for empty text, or text or options
    /// the engine refuses (see [`TtsEngine::start`]); the errors of
    /// [`play`](Self::play).
    pub fn speak(
        &self,
        engine: &TtsEngine,
        text: &str,
        options: TtsOptions,
    ) -> Result<Playback, SpeechError> {
        if text.trim().is_empty() {
            return Err(SpeechError::InvalidInput(
                "there is no text to speak".into(),
            ));
        }
        let (session, playback) = self.speak_streaming(engine, options)?;
        session.push_text(text)?;
        session.close_text();
        Ok(playback)
    }

    /// Starts a synthesis on `engine` and plays it as the audio arrives,
    /// for text that is still being written: push it into the returned
    /// session as it comes, then close it. Returns at once; text pushed
    /// before the synthesis has opened waits for it. Stopping the playback
    /// cancels the synthesis, so later pushes fail.
    ///
    /// # Errors
    ///
    /// The errors [`TtsEngine::start`] returns at once for the options,
    /// and those of [`play`](Self::play).
    pub fn speak_streaming(
        &self,
        engine: &TtsEngine,
        options: TtsOptions,
    ) -> Result<(TtsSession, Playback), SpeechError> {
        let (session, output) =
            engine.start_with(options, Deadline::from(Duration::MAX), Opening::Background)?;
        let playback = self.play(output)?;
        Ok((session, playback))
    }

    /// Queues `sound`: a synthesis's [`TtsOutput`](crate::tts::TtsOutput)
    /// or an [`AudioBuffer`](crate::AudioBuffer). It plays after the sounds queued before it.
    ///
    /// # Errors
    ///
    /// A retryable backend error if the device is lost or its stream
    /// cannot start. The sound is then dropped, which cancels a synthesis.
    pub fn play(&self, sound: impl Into<Sound>) -> Result<Playback, SpeechError> {
        let sound = sound.into();
        if self.shared.device_lost() {
            return Err(lost_error());
        }
        self.start_stream()?;
        let state = Arc::new(PlaybackState::new(sound.text()));
        let mut queue = lock(&self.shared.queue);
        queue.waiting.push_back((state.clone(), sound));
        drop(queue);
        self.shared.changed.notify_all();
        Ok(Playback {
            state,
            shared: self.shared.clone(),
        })
    }

    /// Queues samples you push yourself, at `rate`, into the returned
    /// [`Sink`].
    ///
    /// # Errors
    ///
    /// As [`play`](Self::play).
    pub fn sink(&self, rate: SampleRate) -> Result<(Sink, Playback), SpeechError> {
        let channel = Arc::new(Channel::new(rate));
        let playback = self.play(Sound(Source::Sink(channel.clone())))?;
        Ok((Sink::new(channel), playback))
    }

    /// Stops what is playing and everything queued, and returns at once.
    /// Their synthesis is cancelled, and their `finish` returns `Ok`.
    pub fn stop(&self) {
        let mut queue = lock(&self.shared.queue);
        for (state, sound) in queue.waiting.drain(..) {
            state.settle(Ok(()), Some(0));
            drop(sound);
        }
        queue.stop_all = true;
        drop(queue);
        self.shared.changed.notify_all();
    }

    /// Whether the device stopped. Every queued playback then fails.
    pub fn device_lost(&self) -> bool {
        self.shared.device_lost()
    }

    /// How many times the device ran out of audio while a sound was
    /// playing, and played silence instead.
    pub fn underruns(&self) -> u64 {
        self.shared.underruns.load(Ordering::Relaxed)
    }

    /// Builds and starts the output stream, the first time.
    fn start_stream(&self) -> Result<(), SpeechError> {
        let Output::Device(output) = &self.output else {
            return Ok(());
        };
        let DeviceOutput {
            device,
            config,
            consumer,
            stream,
        } = &**output;
        let mut stream = lock(stream);
        if stream.is_some() {
            return Ok(());
        }
        let Some(consumer) = lock(consumer).take() else {
            return Err(device_error("the output stream could not start"));
        };
        let built = match config.sample_format() {
            cpal::SampleFormat::I16 => build::<i16>(
                device,
                &config.config(),
                self.channels,
                consumer,
                &self.shared,
            ),
            cpal::SampleFormat::U16 => build::<u16>(
                device,
                &config.config(),
                self.channels,
                consumer,
                &self.shared,
            ),
            _ => build::<f32>(
                device,
                &config.config(),
                self.channels,
                consumer,
                &self.shared,
            ),
        }?;
        built.play().map_err(device_error)?;
        *stream = Some(built);
        Ok(())
    }
}

impl Drop for Speaker {
    fn drop(&mut self) {
        self.shared.closing.store(true, Ordering::Release);
        self.shared.changed.notify_all();
        if let Some(player) = self.player.take() {
            let _ = player.join();
        }
        if let Output::Device(output) = &self.output
            && let Some(stream) = lock(&output.stream).take()
        {
            let _ = stream.pause();
        }
    }
}

impl std::fmt::Debug for Speaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Speaker")
            .field("name", &self.name)
            .field("sample_rate", &self.shared.rate)
            .field("device_lost", &self.device_lost())
            .finish_non_exhaustive()
    }
}

fn ring(rate: SampleRate) -> (HeapProd<f32>, HeapCons<f32>) {
    let capacity = usize::try_from(rate.frames_in(BUFFER)).unwrap_or(24_000);
    HeapRb::<f32>::new(capacity.max(1)).split()
}

fn build<T: OutputSample + cpal::SizedSample>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    mut consumer: HeapCons<f32>,
    shared: &Arc<Shared>,
) -> Result<cpal::Stream, SpeechError> {
    let data_shared = shared.clone();
    let error_shared = shared.clone();
    let mut skips: Skips = [0; 2 * SKIPS];
    device
        .build_output_stream(
            *config,
            // Only lock-free pops, atomics, and conversion happen here.
            move |out: &mut [T], info: &cpal::OutputCallbackInfo| {
                let timestamp = info.timestamp();
                let latency = timestamp.playback.duration_since(timestamp.callback);
                data_shared.render(
                    &mut consumer,
                    out,
                    channels,
                    latency,
                    Duration::ZERO,
                    &mut skips,
                );
            },
            move |_: cpal::Error| error_shared.lost.store(true, Ordering::Relaxed),
            None,
        )
        .map_err(device_error)
}

/// How a playback is doing.
struct Progress {
    /// The index of its first sample, once the player has started it.
    start: Option<u64>,
    /// The index after its last sample, once the player has queued it all.
    end: Option<u64>,
    /// Where the device had played to when it was stopped.
    frozen: Option<u64>,
    /// The index where each mark's audio ends, with the text it ends.
    marks: Vec<(u64, usize)>,
    stop: bool,
    /// How the sound ended, reported once it has played out.
    ending: Option<Result<(), SpeechError>>,
    result: Option<Result<(), SpeechError>>,
}

pub(crate) struct PlaybackState {
    text: Option<PushedText>,
    progress: Mutex<Progress>,
    changed: Condvar,
}

impl PlaybackState {
    fn new(text: Option<PushedText>) -> Self {
        Self {
            text,
            progress: Mutex::new(Progress {
                start: None,
                end: None,
                frozen: None,
                marks: Vec::new(),
                stop: false,
                ending: None,
                result: None,
            }),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Progress> {
        lock(&self.progress)
    }

    /// Ends the playback with `result`, having played up to `frozen`, if
    /// it has not ended yet.
    fn settle(&self, result: Result<(), SpeechError>, frozen: Option<u64>) {
        let mut progress = self.lock();
        Self::settle_locked(&mut progress, result, frozen);
        drop(progress);
        self.changed.notify_all();
    }

    /// [`settle`](Self::settle) for a caller that holds the lock.
    fn settle_locked(
        progress: &mut Progress,
        result: Result<(), SpeechError>,
        frozen: Option<u64>,
    ) {
        if progress.result.is_none() {
            progress.result = Some(result);
            if progress.frozen.is_none() {
                progress.frozen = frozen;
            }
        }
    }

    fn is_stopped(&self) -> bool {
        self.lock().stop
    }
}

/// One queued sound, from a [`Speaker`].
///
/// Follow it with [`played`](Self::played) and
/// [`text_played`](Self::text_played), skip it with [`stop`](Self::stop),
/// or wait for it with [`finish`](Self::finish). It is `Send + Sync`.
/// Dropping it lets it play.
pub struct Playback {
    state: Arc<PlaybackState>,
    shared: Arc<Shared>,
}

impl Playback {
    /// Skips the sound, and returns at once: what is queued of it is not
    /// played, and a synthesis it plays is cancelled. Stopping is not an
    /// error: [`finish`](Self::finish) then returns `Ok`.
    pub fn stop(&self) {
        self.halt(Ok(()));
    }

    /// Stops the playback with `result`, unless it already ended.
    fn halt(&self, result: Result<(), SpeechError>) {
        let played = self.shared.played_index();
        // A sound still queued is dropped here, which cancels a synthesis;
        // the player skips the rest of one it has started. The queue is
        // locked before the progress, as `Speaker::stop` does.
        let removed = {
            let mut queue = lock(&self.shared.queue);
            let waiting = queue
                .waiting
                .iter()
                .position(|(state, _)| Arc::ptr_eq(state, &self.state));
            waiting.and_then(|index| queue.waiting.remove(index))
        };
        let mut progress = self.state.lock();
        if progress.result.is_some() || progress.stop {
            return;
        }
        // The stop and its result are set together: the player settles a
        // stopped sound with `Ok`, and must never do that before this
        // result is in, or a `finish` that timed out would report success.
        progress.stop = true;
        let frozen = if removed.is_some() { 0 } else { played };
        PlaybackState::settle_locked(&mut progress, result, Some(frozen));
        drop(progress);
        self.state.changed.notify_all();
        drop(removed);
        self.shared.changed.notify_all();
    }

    /// Waits until the sound has played to its end or was stopped.
    ///
    /// # Errors
    ///
    /// A retryable backend error if the device is lost; the synthesis's
    /// error, after playing what arrived; [`SpeechError::DeadlineExceeded`]
    /// if `deadline` passes, which stops the playback and cancels its
    /// synthesis; [`SpeechError::Closed`] if the speaker was dropped.
    pub fn finish(&self, deadline: impl Into<Deadline>) -> Result<(), SpeechError> {
        let deadline = deadline.into();
        let state = &*self.state;
        let (progress, done) = wait_until(&state.changed, state.lock(), deadline, |progress| {
            progress.result.is_some()
        });
        drop(progress);
        if !done {
            self.halt(Err(SpeechError::DeadlineExceeded));
        }
        self.state
            .lock()
            .result
            .clone()
            .unwrap_or(Err(SpeechError::Closed))
    }

    /// Whether the sound has played to its end, was stopped, or failed.
    pub fn is_done(&self) -> bool {
        self.state.lock().result.is_some()
    }

    /// The audio of this sound the device has played: only samples whose
    /// playback time has passed count (D-06).
    pub fn played(&self) -> Duration {
        let progress = self.state.lock();
        let frames = self.played_frames(&progress);
        self.shared.rate.duration_of(frames)
    }

    /// The text whose audio has fully played: what the listener heard, to
    /// the nearest mark, typically a sentence. Empty for a sound that is
    /// not a synthesis.
    pub fn text_played(&self) -> String {
        let Some(text) = &self.state.text else {
            return String::new();
        };
        let progress = self.state.lock();
        let Some(start) = progress.start else {
            return String::new();
        };
        let upto = start + self.played_frames(&progress);
        let end = progress
            .marks
            .iter()
            .take_while(|(index, _)| *index <= upto)
            .map(|(_, end)| *end)
            .max()
            .unwrap_or(0);
        drop(progress);
        text.prefix(end)
    }

    fn played_frames(&self, progress: &Progress) -> u64 {
        let Some(start) = progress.start else {
            return 0;
        };
        let played = progress
            .frozen
            .unwrap_or_else(|| self.shared.played_index());
        let played = progress.end.map_or(played, |end| played.min(end));
        played.saturating_sub(start)
    }
}

impl std::fmt::Debug for Playback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Playback")
            .field("played", &self.played())
            .field("done", &self.is_done())
            .finish_non_exhaustive()
    }
}

/// A sound being queued into the ring.
struct Current {
    state: Arc<PlaybackState>,
    sound: Sound,
    resampler: Resampler,
    /// Samples at the device's rate not yet in the ring.
    pending: Vec<f32>,
    /// How far into an `AudioBuffer` the player has got.
    offset: usize,
    /// No more audio will come from the sound.
    ended: Option<Result<(), SpeechError>>,
}

/// The player thread's state.
struct Player {
    shared: Arc<Shared>,
    producer: HeapProd<f32>,
    /// Samples written to the ring so far: the next sample's index.
    pushed: u64,
    current: Option<Current>,
    /// Sounds fully in the ring, waiting to play out, with their end.
    tail: VecDeque<Arc<PlaybackState>>,
    /// Ranges of stopped sounds the callback has not passed yet, in index
    /// order; the first [`SKIPS`] are published.
    skips: Vec<(u64, u64)>,
    published: Vec<(u64, u64)>,
    reported_underruns: u64,
}

impl Player {
    fn new(shared: Arc<Shared>, producer: HeapProd<f32>) -> Self {
        Self {
            shared,
            producer,
            pushed: 0,
            current: None,
            tail: VecDeque::new(),
            skips: Vec::new(),
            published: Vec::new(),
            reported_underruns: 0,
        }
    }

    fn run(mut self) {
        loop {
            #[cfg(test)]
            self.shared.loops.fetch_add(1, Ordering::Relaxed);
            if self.shared.closing.load(Ordering::Acquire) {
                self.fail_all(&SpeechError::Closed);
                return;
            }
            if self.shared.device_lost() {
                tracing::warn!("the speaker stopped; its playbacks fail");
                self.fail_all(&lost_error());
                self.wait();
                continue;
            }
            self.service();
            if self.current.is_none() {
                self.next_sound();
            }
            let progressed = self.current.is_some() && self.feed();
            if !progressed {
                self.wait();
            }
        }
    }

    /// Waits a tick, or until something is queued or stopped. A queued
    /// sound is work only while none is playing: behind a playing one it
    /// waits its turn, so the player sleeps as usual, and does not spin for
    /// as long as the first sound plays.
    fn wait(&self) {
        let queue = lock(&self.shared.queue);
        let work = self.current.is_none() && !queue.waiting.is_empty();
        if !work && !queue.stop_all {
            let _ = self
                .shared
                .changed
                .wait_timeout(queue, TICK)
                .map_err(|_| ());
        }
    }

    /// Handles stops, playbacks that have played out, skips, and underruns.
    fn service(&mut self) {
        let stop_all = std::mem::take(&mut lock(&self.shared.queue).stop_all);
        let played = self.shared.played_index();
        if stop_all || self.current.as_ref().is_some_and(|c| c.state.is_stopped()) {
            if let Some(current) = self.current.take() {
                self.skip_rest(&current.state, played);
                current.state.settle(Ok(()), Some(played));
                if let Source::Sink(channel) = &current.sound.0 {
                    channel.end();
                }
                // Dropping the sound cancels its synthesis.
                drop(current);
            }
            self.shared.expecting.store(false, Ordering::Relaxed);
        }
        let mut tail = std::mem::take(&mut self.tail);
        tail.retain(|state| {
            if stop_all || state.is_stopped() {
                self.skip_rest(state, played);
                state.settle(Ok(()), Some(played));
                return false;
            }
            let progress = state.lock();
            let end = progress.end.unwrap_or(u64::MAX);
            if played < end {
                return true;
            }
            drop(progress);
            // Its outcome waits until it has played out.
            let ending = state.lock().ending.take().unwrap_or(Ok(()));
            state.settle(ending, None);
            false
        });
        self.tail = tail;
        self.publish_skips();
        let underruns = self.shared.underruns.load(Ordering::Relaxed);
        if underruns > self.reported_underruns {
            tracing::warn!(
                underruns = underruns - self.reported_underruns,
                "the speaker ran out of audio and played silence"
            );
            self.reported_underruns = underruns;
        }
    }

    /// Skips what is in the ring of a stopped playback.
    fn skip_rest(&mut self, state: &PlaybackState, played: u64) {
        let progress = state.lock();
        let Some(start) = progress.start else {
            return;
        };
        let end = progress.end.unwrap_or(self.pushed);
        drop(progress);
        let from = start
            .max(played)
            .max(self.shared.popped.load(Ordering::Relaxed));
        if from < end {
            self.skips.push((from, end));
        }
        self.publish_skips();
    }

    /// Drops the ranges the callback has passed, and publishes the first
    /// [`SKIPS`] of the rest if they changed.
    fn publish_skips(&mut self) {
        let popped = self.shared.popped.load(Ordering::Relaxed);
        self.skips.retain(|&(_, to)| to > popped);
        self.skips.sort_unstable();
        let mut merged: Vec<(u64, u64)> = Vec::with_capacity(self.skips.len());
        for &(from, to) in &self.skips {
            match merged.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => merged.push((from, to)),
            }
        }
        self.skips = merged;
        let first: Vec<(u64, u64)> = self.skips.iter().copied().take(SKIPS).collect();
        if first != self.published {
            let mut values = [0; 2 * SKIPS];
            for (slot, &(from, to)) in values.as_chunks_mut::<2>().0.iter_mut().zip(&first) {
                *slot = [from, to];
            }
            self.shared.skip.write(values);
            self.published = first;
        }
    }

    fn next_sound(&mut self) {
        let Some((state, sound)) = lock(&self.shared.queue).waiting.pop_front() else {
            return;
        };
        match Resampler::new(sound.sample_rate(), self.shared.rate) {
            Ok(resampler) => {
                self.current = Some(Current {
                    state,
                    sound,
                    resampler,
                    pending: Vec::new(),
                    offset: 0,
                    ended: None,
                });
            }
            Err(error) => state.settle(Err(error), Some(0)),
        }
    }

    /// Moves audio from the current sound into the ring. Returns whether
    /// anything moved.
    fn feed(&mut self) -> bool {
        let Some(current) = self.current.as_mut() else {
            return false;
        };
        if !current.pending.is_empty() {
            let pushed = self.producer.push_slice(&current.pending);
            if pushed > 0 {
                let mut progress = current.state.lock();
                progress.start.get_or_insert(self.pushed);
                drop(progress);
                current.pending.drain(..pushed);
                self.pushed += pushed as u64;
                let prefill = self.shared.rate.frames_in(PREFILL);
                let start = current.state.lock().start.unwrap_or(self.pushed);
                if self.pushed - start >= prefill {
                    self.shared.expecting.store(true, Ordering::Relaxed);
                }
            }
            return pushed > 0;
        }
        if let Some(result) = current.ended.take() {
            let mut progress = current.state.lock();
            let start = *progress.start.get_or_insert(self.pushed);
            progress.end = Some(self.pushed);
            progress.ending = Some(result);
            drop(progress);
            self.shared.expecting.store(false, Ordering::Relaxed);
            let state = current.state.clone();
            self.current = None;
            if self.pushed == start {
                let ending = state.lock().ending.take().unwrap_or(Ok(()));
                state.settle(ending, None);
            } else {
                self.tail.push_back(state);
            }
            return true;
        }
        let rate = self.shared.rate;
        let piece = usize::try_from(current.sound.sample_rate().frames_in(PIECE)).unwrap_or(1_600);
        let mut input = Vec::new();
        let mut finished: Option<Result<(), SpeechError>> = None;
        let start = current.state.lock().start;
        match &mut current.sound.0 {
            Source::Tts(output) => match output.recv(Deadline::from(TICK)) {
                Ok(TtsUpdate::Audio(samples)) => input = samples,
                Ok(TtsUpdate::Mark(mark)) => {
                    let base = start.unwrap_or(self.pushed);
                    let index = base + rate.frames_in(mark.audio.end);
                    current.state.lock().marks.push((index, mark.text.end));
                    return true;
                }
                Ok(TtsUpdate::Closed(Ok(_))) | Err(RecvError::Closed) => finished = Some(Ok(())),
                Ok(TtsUpdate::Closed(Err(failure))) => finished = Some(Err(failure.error)),
                Err(RecvError::Timeout | RecvError::Empty) => return false,
            },
            Source::Buffer(buffer) => {
                let end = (current.offset + piece).min(buffer.samples.len());
                input.extend_from_slice(&buffer.samples[current.offset..end]);
                current.offset = end;
                if end == buffer.samples.len() {
                    finished = Some(Ok(()));
                }
            }
            Source::Sink(channel) => match channel.take(piece, Deadline::from(TICK)) {
                Taken::Audio(samples) => input = samples,
                Taken::Nothing => return false,
                Taken::End => finished = Some(Ok(())),
            },
        }
        let mut resampled = Vec::with_capacity(input.len());
        let mut outcome = current.resampler.process(&input, &mut resampled);
        if finished.is_some() && outcome.is_ok() {
            outcome = current.resampler.flush(&mut resampled);
        }
        current.pending = resampled;
        match outcome {
            Ok(()) => current.ended = finished,
            Err(error) => current.ended = Some(Err(error)),
        }
        true
    }

    /// Fails every playback, as when the device is lost or the speaker is
    /// dropped.
    fn fail_all(&mut self, error: &SpeechError) {
        let played = self.shared.played_index();
        let waiting: Vec<_> = lock(&self.shared.queue).waiting.drain(..).collect();
        for (state, _sound) in waiting {
            state.settle(Err(error.clone()), Some(0));
        }
        if let Some(current) = self.current.take() {
            current.state.settle(Err(error.clone()), Some(played));
            if let Source::Sink(channel) = &current.sound.0 {
                channel.end();
            }
        }
        for state in self.tail.drain(..) {
            state.settle(Err(error.clone()), Some(played));
        }
        self.shared.expecting.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests;
