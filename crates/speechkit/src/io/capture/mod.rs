//! Microphone capture with cpal: one timeline per running microphone, read
//! by listenings and watchings.

use std::{
    sync::{
        Arc, Mutex, MutexGuard, PoisonError, Weak,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    },
    time::Duration,
};

use cpal::traits::{DeviceTrait, StreamTrait};
use ringbuf::{
    HeapProd, HeapRb,
    traits::{Consumer, Producer, Split},
};

use crate::{
    SampleRate, SpeechError,
    asr::{AsrEngine, AsrOptions},
    io::{
        DeviceInfo, Opened,
        convert::{DeviceSample, mixdown},
        list, open,
    },
    wake::WakeWordModel,
};

mod fake;
mod listening;
mod timeline;
mod watching;

pub use fake::FakeMicrophone;
pub use listening::{ListenOptions, Listening, Recording};
pub use watching::{Wake, WakeUpdate, WatchOptions, Watching};

use timeline::Timeline;

/// The ring buffer between the device callback and the capture thread
/// holds this much audio.
const BUFFER: Duration = Duration::from_secs(2);
/// The capture thread wakes this often.
const TICK: Duration = Duration::from_millis(20);
/// A reservation holds the audio after a wake word for this long (D-03).
const RESERVATION: Duration = Duration::from_secs(30);
/// [`Device::lost_at`] when no samples were lost since the capture thread
/// last looked.
const NOTHING_LOST: u64 = u64::MAX;

fn device_error(error: impl std::fmt::Display) -> SpeechError {
    SpeechError::backend("microphone", true, error.to_string())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Frames in `duration` at `rate`, saturating.
fn frames(rate: SampleRate, duration: Duration) -> u64 {
    rate.frames_in(duration)
}

/// Where a microphone's audio comes from.
enum Source {
    Device {
        device: cpal::Device,
        config: cpal::SupportedStreamConfig,
    },
    Fake(Arc<fake::Inner>),
}

/// An input device, ready to capture.
///
/// Open it with [`open_default`](Self::open_default), or by name with
/// [`open`](Self::open). [`capture`](Self::capture) starts it and returns
/// a [`Capture`], which listenings and wake-word watchings read.
/// [`listen`](Self::listen) is the short way to one request: it captures,
/// then listens from the start.
pub struct Microphone {
    source: Source,
    name: String,
    sample_rate: SampleRate,
}

impl Microphone {
    /// The system's default input device.
    ///
    /// # Errors
    ///
    /// A retryable backend error if there is no input device or it has no
    /// usable configuration, or `Unsupported` for its sample format.
    pub fn open_default() -> Result<Self, SpeechError> {
        Ok(Self::from_device(open(true, None)?))
    }

    /// The input device called `name`, as [`list`](Self::list) lists it.
    /// The exact name wins, then the name ignoring case, then a part of one
    /// name ignoring case.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] if `name` is empty, matches no input
    /// device, or matches several, including devices with the same name;
    /// a retryable backend error if the devices cannot be listed or the
    /// device has no usable configuration; or `Unsupported` for its sample
    /// format.
    pub fn open(name: &str) -> Result<Self, SpeechError> {
        Ok(Self::from_device(open(true, Some(name))?))
    }

    /// The input devices, in the order the system lists them.
    ///
    /// # Errors
    ///
    /// A retryable backend error if the devices cannot be listed.
    pub fn list() -> Result<Vec<DeviceInfo>, SpeechError> {
        list(true)
    }

    /// A microphone with no device, and the handle a test feeds it
    /// through. For the device contract tests; not covered by semver.
    #[doc(hidden)]
    pub fn fake(sample_rate: SampleRate) -> (Self, FakeMicrophone) {
        let inner = Arc::new(fake::Inner::default());
        let microphone = Self {
            source: Source::Fake(inner.clone()),
            name: "fake".into(),
            sample_rate,
        };
        (microphone, FakeMicrophone { inner })
    }

    fn from_device(opened: Opened) -> Self {
        Self {
            source: Source::Device {
                device: opened.device,
                config: opened.config,
            },
            name: opened.name,
            sample_rate: opened.rate,
        }
    }

    /// The device's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The rate of the mono audio the microphone delivers.
    pub fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// Starts the microphone. It runs while the [`Capture`], one of its
    /// clones, or an unfinished listening or watching on it exists, or
    /// until [`Capture::stop`].
    ///
    /// # Errors
    ///
    /// A retryable backend error if the device cannot start.
    pub fn capture(&self, options: CaptureOptions) -> Result<Capture, SpeechError> {
        let rate = self.sample_rate;
        let device = Arc::new(Device::new(rate, &options));
        let capacity = usize::try_from(frames(rate, BUFFER)).unwrap_or(96_000);
        let (producer, consumer) = HeapRb::<f32>::new(capacity).split();
        match &self.source {
            Source::Device {
                device: cpal,
                config,
            } => {
                let channels = usize::from(config.channels());
                let stream_config = config.config();
                let stream = match config.sample_format() {
                    cpal::SampleFormat::I16 => {
                        build::<i16>(cpal, &stream_config, channels, producer, &device)
                    }
                    cpal::SampleFormat::U16 => {
                        build::<u16>(cpal, &stream_config, channels, producer, &device)
                    }
                    _ => build::<f32>(cpal, &stream_config, channels, producer, &device),
                }?;
                stream.play().map_err(device_error)?;
                *lock(&device.stream) = Some(stream);
            }
            Source::Fake(inner) => inner.attach(&device, producer),
        }
        let thread_device = device.clone();
        std::thread::Builder::new()
            .name("speechkit-capture".into())
            .spawn(move || run(&thread_device, consumer))
            .map_err(|error| {
                device.halt();
                device_error(error)
            })?;
        Ok(Capture {
            hold: Arc::new(Hold(device.clone())),
            device,
        })
    }

    /// Captures, then listens from the start, as one request: the short
    /// way to [`Capture::listen`]. The listening is the capture's only
    /// user, so stopping it stops the microphone.
    ///
    /// # Errors
    ///
    /// The errors of [`capture`](Self::capture) and [`Capture::listen`].
    pub fn listen(
        &self,
        engine: &AsrEngine,
        options: AsrOptions,
    ) -> Result<Listening, SpeechError> {
        self.capture(CaptureOptions::default())?
            .listen(engine, options, ListenOptions::default())
    }
}

impl std::fmt::Debug for Microphone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Microphone")
            .field("name", &self.name)
            .field("sample_rate", &self.sample_rate)
            .finish_non_exhaustive()
    }
}

/// How a [`Capture`] holds audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CaptureOptions {
    /// How far back a listening may start, with
    /// [`ListenOptions::starting_at`]. Default: 5 s, about 1 MB at 48 kHz.
    pub history: Duration,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            history: Duration::from_secs(5),
        }
    }
}

impl CaptureOptions {
    /// Sets [`history`](Self::history).
    #[must_use]
    pub const fn with_history(mut self, history: Duration) -> Self {
        self.history = history;
        self
    }
}

/// A running microphone, shared by the capture's handles, its readers,
/// the callback, and the capture thread.
pub(crate) struct Device {
    rate: SampleRate,
    timeline: Timeline,
    /// The RMS level of the latest device buffer, as `f32` bits.
    level: AtomicU32,
    lost: AtomicBool,
    stopping: AtomicBool,
    /// Samples the callback could not queue.
    dropped: AtomicU64,
    /// Samples the callback queued, which is where the next one lands in
    /// the timeline. Only the callback writes it.
    queued: AtomicU64,
    /// Where in the timeline the latest samples were lost: the position of
    /// the first sample queued after them, or [`NOTHING_LOST`]. The
    /// callback stores it before it queues anything after the loss.
    lost_at: AtomicU64,
    stream: Mutex<Option<cpal::Stream>>,
}

impl Device {
    fn new(rate: SampleRate, options: &CaptureOptions) -> Self {
        Self {
            rate,
            timeline: Timeline::new(frames(rate, options.history), frames(rate, RESERVATION)),
            level: AtomicU32::new(0),
            lost: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            dropped: AtomicU64::new(0),
            queued: AtomicU64::new(0),
            lost_at: AtomicU64::new(NOTHING_LOST),
            stream: Mutex::new(None),
        }
    }

    /// Stops the device. The capture thread then moves the audio still in
    /// the ring buffer and ends the timeline. Returns at once.
    fn halt(&self) {
        // Release pairs with the capture thread's Acquire load, after the
        // device's last buffer.
        self.stopping.store(true, Ordering::Release);
        let stream = lock(&self.stream).take();
        if let Some(stream) = stream {
            let _ = stream.pause();
        }
        self.level.store(0, Ordering::Relaxed);
    }

    fn device_lost(&self) -> bool {
        self.lost.load(Ordering::Relaxed)
    }

    fn level(&self) -> f32 {
        if self.device_lost() || self.stopping.load(Ordering::Relaxed) {
            return 0.0;
        }
        f32::from_bits(self.level.load(Ordering::Relaxed))
    }

    fn position(&self) -> Duration {
        self.rate.duration_of(self.timeline.position())
    }

    /// Takes what a device callback delivered: converts it to mono, queues
    /// it, and measures its level. What does not fit in the queue is lost,
    /// and where is recorded, for the timeline. Only arithmetic, atomics,
    /// and a lock-free push happen here.
    fn input<T: DeviceSample>(
        &self,
        data: &[T],
        channels: usize,
        producer: &mut impl Producer<Item = f32>,
    ) {
        let (mut dropped, mut squares, mut count) = (0_u64, 0.0_f32, 0.0_f32);
        let mut queued = self.queued.load(Ordering::Relaxed);
        let mut losing = false;
        mixdown(data, channels, |sample| {
            squares += sample * sample;
            count += 1.0;
            if producer.try_push(sample).is_ok() {
                queued += 1;
                losing = false;
            } else {
                dropped += 1;
                if !losing {
                    losing = true;
                    // Stored here, not after the loop: the capture thread
                    // can empty the ring at any moment, and the audio it
                    // then moves from after this loss must come with it.
                    // The latest loss wins: a reader that would read
                    // across an earlier one would read across this one
                    // too. Release pairs with the capture thread's Acquire
                    // swap, which follows its Acquire read of the samples
                    // queued after this store.
                    self.lost_at.store(queued, Ordering::Release);
                }
            }
        });
        if count > 0.0 {
            let rms = (squares / count).sqrt();
            self.level.store(rms.to_bits(), Ordering::Relaxed);
        }
        self.queued.store(queued, Ordering::Relaxed);
        if dropped > 0 {
            self.dropped.fetch_add(dropped, Ordering::Relaxed);
        }
    }
}

fn build<T: DeviceSample + cpal::SizedSample>(
    cpal: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    mut producer: HeapProd<f32>,
    device: &Arc<Device>,
) -> Result<cpal::Stream, SpeechError> {
    // Weak, so a stream the host keeps alive does not keep the device.
    let data_device = Arc::downgrade(device);
    let error_device = Arc::downgrade(device);
    cpal.build_input_stream(
        *config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            if let Some(device) = data_device.upgrade() {
                device.input(data, channels, &mut producer);
            }
        },
        move |_: cpal::Error| {
            if let Some(device) = error_device.upgrade() {
                device.lost.store(true, Ordering::Release);
            }
        },
        None,
    )
    .map_err(device_error)
}

/// The capture thread: moves the audio from the ring buffer into the
/// timeline every tick, until the device stops or is lost.
fn run(device: &Device, mut consumer: impl Consumer<Item = f32>) {
    let mut moved = Vec::new();
    loop {
        std::thread::sleep(TICK);
        // Acquire pairs with the stores in `halt` and the error callback,
        // so the audio the device delivered before them is seen below.
        let stopping = device.stopping.load(Ordering::Acquire);
        let lost = device.lost.load(Ordering::Acquire);
        moved.resize(consumer.occupied_len(), 0.0);
        let count = consumer.pop_slice(&mut moved);
        moved.truncate(count);
        // Read after the ring was emptied: a loss is stored before the
        // callback queues any sample after it, so if what was just moved
        // includes a sample from after a loss, the loss is seen here.
        let lost_at = device.lost_at.swap(NOTHING_LOST, Ordering::Acquire);
        if lost_at == NOTHING_LOST {
            device.timeline.append(&moved);
        } else {
            device.timeline.append_with_loss(&moved, lost_at);
        }
        let dropped = device.dropped.swap(0, Ordering::Relaxed);
        if dropped > 0 {
            tracing::warn!(
                dropped,
                "the microphone's buffer overflowed and samples were lost, so readers that had not got to them fail; the machine is too slow"
            );
        }
        if lost {
            tracing::warn!("the microphone stopped delivering audio; its readers end here");
            device.halt();
        }
        if stopping || lost {
            device.timeline.end();
            return;
        }
    }
}

/// Keeps the device running while it exists. The last one stops it.
pub(crate) struct Hold(Arc<Device>);

impl Drop for Hold {
    fn drop(&mut self) {
        self.0.halt();
    }
}

/// A running microphone, from [`Microphone::capture`].
///
/// A capture has one timeline: capture time, the audio captured since it
/// started. Readers start at a position it still holds, in its history
/// (see [`CaptureOptions::history`]) or in a wake-word reservation, and
/// each receives every sample from there on, in order (D-01):
///
/// - [`listen`](Self::listen) starts one request, whose session opens in
///   the background while the listening holds the audio;
/// - [`watch`](Self::watch) runs a wake-word detector on its own thread.
///
/// The microphone runs while a `Capture` (it is `Clone`), or a listening
/// or watching that has not ended, exists. [`stop`](Self::stop) stops it
/// for every reader.
///
/// The microphone's buffer overflows if the machine falls behind by more
/// than 2 s. Its readers never skip the samples lost: each that had not yet
/// read past them fails with [`SpeechError::Capacity`], and none can start
/// before them (D-02). Capture time counts the audio delivered, so the
/// time after a loss is early by the audio lost. The loss is logged at
/// `warn`, as is a lost device, which ends every reader at its last sample
/// (D-05).
#[derive(Clone)]
pub struct Capture {
    device: Arc<Device>,
    hold: Arc<Hold>,
}

impl Capture {
    /// The rate of the audio captured.
    pub fn sample_rate(&self) -> SampleRate {
        self.device.rate
    }

    /// Capture time: the audio captured so far.
    pub fn position(&self) -> Duration {
        self.device.position()
    }

    /// The level of the latest audio, as the RMS of the samples of the
    /// device's latest buffer, from 0.0 to 1.0: typically 5 to 20 ms of
    /// audio, for drawing a live meter. Zero once stopped or once the
    /// device is lost.
    pub fn level(&self) -> f32 {
        self.device.level()
    }

    /// Whether the device stopped delivering audio before it was stopped.
    /// It turns true as soon as the device reports the failure.
    pub fn device_lost(&self) -> bool {
        self.device.device_lost()
    }

    /// Starts one request: a listening that feeds a session on `engine`
    /// from the position `listen` names. It checks the options at once and
    /// returns; the session opens in the background (waiting for a slot,
    /// loading, or connecting) while the listening holds the audio, and
    /// then receives the held audio first and live audio after it. Its
    /// times are capture times.
    ///
    /// # Errors
    ///
    /// - [`SpeechError::Capacity`] if the start is no longer held (D-01);
    /// - [`SpeechError::InvalidInput`] if the options are invalid, or the
    ///   start is later than [`position`](Self::position);
    /// - [`SpeechError::Closed`] once the capture has stopped;
    /// - the errors [`AsrEngine::start`] returns at once for the options.
    pub fn listen(
        &self,
        engine: &AsrEngine,
        options: AsrOptions,
        listen: ListenOptions,
    ) -> Result<Listening, SpeechError> {
        listen.validate()?;
        let rate = self.device.rate;
        let start = listen.start.map(|start| frames(rate, start));
        let limit = frames(rate, listen.max_backlog);
        let (reader, start) = self.device.timeline.add_reader(start, 0, limit)?;
        Listening::start(
            &self.device,
            self.hold.clone(),
            reader,
            start,
            engine,
            options,
            &listen,
        )
    }

    /// Runs a detector from `model` on the audio from now on, on its own
    /// thread. When it hears a keyword, the audio after the keyword is
    /// reserved at once, before the application reads the event (D-03), so
    /// [`Wake::listen`] starts right after the keyword. The default
    /// [`WatchOptions`] apply; [`watch_with`](Self::watch_with) sets them.
    ///
    /// # Errors
    ///
    /// [`SpeechError::Closed`] once the capture has stopped, or an error
    /// creating the detector.
    pub fn watch(&self, model: &dyn WakeWordModel) -> Result<Watching, SpeechError> {
        self.watch_with(model, WatchOptions::default())
    }

    /// [`watch`](Self::watch), with `options`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] if the options are invalid,
    /// [`SpeechError::Closed`] once the capture has stopped, or an error
    /// creating the detector.
    pub fn watch_with(
        &self,
        model: &dyn WakeWordModel,
        options: WatchOptions,
    ) -> Result<Watching, SpeechError> {
        Watching::start(&self.device, &self.hold, model, options)
    }

    /// Stops the microphone for every reader, and returns at once. Every
    /// listening then ends at the last sample, and its session finishes
    /// with what it heard; every watching ends too (D-04). Calling it
    /// again does nothing.
    pub fn stop(&self) {
        self.device.halt();
    }
}

impl std::fmt::Debug for Capture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Capture")
            .field("sample_rate", &self.device.rate)
            .field("position", &self.position())
            .field("device_lost", &self.device_lost())
            .finish_non_exhaustive()
    }
}

/// A weak reference to a capture's hold, for a [`Wake`], which must not
/// keep the microphone running.
type WeakHold = Weak<Hold>;

#[cfg(test)]
mod tests;
