//! Streaming transducer recognition, such as streaming Zipformer.

use std::{sync::Arc, time::Duration};

use crate::{
    SampleRate, SpeechError,
    asr::{
        AsrBackend, AsrCapabilities, AsrEvent, AsrEvents, AsrOptions, AsrStream, Partial, Segment,
        UtteranceId,
    },
};
use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, OnlineStream};

use super::{RATE, TAIL_PADDING, native};
use crate::sherpa::{
    bias::{self, ModelingUnit, TransducerBias, Vocabulary},
    config::{Inference, ModelFiles},
    layout::AsrFiles,
};

/// Settings for [`SherpaStreaming`], checked by `AsrConfig`.
#[derive(Debug)]
pub(crate) struct StreamingConfig {
    /// The model's files.
    pub files: ModelFiles,
    /// Where and how the model runs.
    pub inference: Inference,
    /// Decoding bias. Setting it, even with no phrases, enables session
    /// hints and switches to modified beam search (2–4× slower).
    pub bias: Option<PreparedBias>,
    /// The pause that ends an utterance.
    pub endpoint_silence: Duration,
    /// Where an unfinished utterance is cut.
    pub max_utterance: Duration,
}

/// How far behind the input the decoder may report the first token of
/// speech: the model's chunk and right context plus the transducer's
/// emission delay. `streaming_activity_lags_the_input_by_at_most_the_latency`
/// prints it: 420 to 460 ms on streaming-en (2023-06-26) in 100 ms blocks.
/// The margin covers models with larger chunks.
pub(crate) const LATENCY: Duration = Duration::from_millis(1_000);

/// A streaming transducer, such as streaming Zipformer.
///
/// It reports a partial result whenever the text changes, and commits a
/// segment at each endpoint and at the end of input. An endpoint is a
/// pause of `endpoint_silence` after decoded speech, or an utterance that
/// reached `max_utterance`, which is a cut.
///
/// Speech starts at an utterance's first token and ends at its last one,
/// reported at the endpoint. Activity is known up to the input minus
/// [`LATENCY`] between speech.
pub(crate) struct SherpaStreaming {
    recognizer: Arc<OnlineRecognizer>,
    caps: AsrCapabilities,
    hotwords: Hotwords,
    endpoint_silence: Duration,
}

/// Bias state kept after loading, for merging session hints.
#[derive(Debug, Default)]
pub(crate) struct Hotwords {
    /// The engine phrases, rendered.
    pub(crate) engine: Option<String>,
    /// Characters the model knows, for checking session hints.
    pub(crate) vocabulary: Vocabulary,
}

impl Hotwords {
    /// The hotwords for a session, or `None` for plain decoding.
    pub(crate) fn for_session(&self, opts: &AsrOptions) -> Result<Option<String>, SpeechError> {
        bias::check_session_hints(&opts.hints)?;
        opts.hints
            .iter()
            .try_for_each(|phrase| self.vocabulary.check(phrase))?;
        bias::merge(self.engine.as_deref(), &opts.hints)
    }
}

/// What configuring `bias` means for a transducer in `model`: the unit,
/// `bpe.vocab`, the boost of phrases without their own, and the checked
/// phrases.
#[derive(Debug)]
pub(crate) struct PreparedBias {
    pub(crate) unit: ModelingUnit,
    pub(crate) bpe_vocab: Option<String>,
    pub(crate) score: f32,
    pub(crate) hotwords: Hotwords,
}

pub(crate) fn prepare_bias(
    bias: &TransducerBias,
    model: &ModelFiles,
) -> Result<PreparedBias, SpeechError> {
    bias.validate()?;
    let AsrFiles::Transducer {
        tokens, bpe_vocab, ..
    } = &model.files
    else {
        return Err(SpeechError::Unsupported(
            "decoding bias needs a transducer model".into(),
        ));
    };
    let text = std::fs::read_to_string(model.path(tokens))
        .map_err(|e| SpeechError::InvalidModel(format!("cannot read {tokens}: {e}")))?;
    let unit = ModelingUnit::infer(bpe_vocab.is_some(), &text);
    let bpe_vocab = if unit.needs_bpe_vocab() {
        let vocab = bpe_vocab.as_ref().ok_or_else(|| {
            SpeechError::InvalidModel(format!(
                "modeling unit {} needs bpe.vocab in the model directory",
                unit.as_str()
            ))
        })?;
        Some(model.path(vocab).to_string_lossy().into_owned())
    } else {
        None
    };
    let vocabulary = Vocabulary::new(unit, &text);
    bias.phrases
        .iter()
        .try_for_each(|phrase| vocabulary.check(&phrase.text))?;
    Ok(PreparedBias {
        unit,
        bpe_vocab,
        score: bias.default_score,
        hotwords: Hotwords {
            engine: bias.render(),
            vocabulary,
        },
    })
}

impl SherpaStreaming {
    /// Loads the model from the files and bias `AsrConfig` checked.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for bad settings, or
    /// [`SpeechError::InvalidModel`] for a model the native library rejects.
    pub(crate) fn load(config: StreamingConfig) -> Result<Self, SpeechError> {
        config.inference.validate()?;
        let model = &config.files;
        let AsrFiles::Transducer {
            encoder,
            decoder,
            joiner,
            tokens,
            ..
        } = &model.files
        else {
            return Err(SpeechError::InvalidModel("not a transducer".into()));
        };
        let path = |name: &String| Some(model.path(name).to_string_lossy().into_owned());
        let mut native_config = OnlineRecognizerConfig::default();
        native_config.model_config.transducer.encoder = path(encoder);
        native_config.model_config.transducer.decoder = path(decoder);
        native_config.model_config.transducer.joiner = path(joiner);
        native_config.model_config.tokens = path(tokens);
        native_config.model_config.num_threads = config.inference.threads_i32();
        native_config.model_config.provider = Some(config.inference.provider.as_str().into());
        let mut hotwords = Hotwords::default();
        let accepts_hints = config.bias.is_some();
        match config.bias {
            Some(prepared) => {
                native_config.decoding_method = Some("modified_beam_search".into());
                native_config.max_active_paths = 4;
                native_config.hotwords_score = prepared.score;
                native_config.model_config.modeling_unit = Some(prepared.unit.as_str().into());
                native_config.model_config.bpe_vocab = prepared.bpe_vocab;
                hotwords = prepared.hotwords;
            }
            None => native_config.decoding_method = Some("greedy_search".into()),
        }
        native_config.enable_endpoint = true;
        // Rule 1 resets a stream that decoded nothing; rule 2 is a pause
        // after speech, and rule 3 a cut.
        native_config.rule1_min_trailing_silence = config.endpoint_silence.as_secs_f32();
        native_config.rule2_min_trailing_silence = config.endpoint_silence.as_secs_f32();
        native_config.rule3_min_utterance_length = config.max_utterance.as_secs_f32();
        let recognizer = OnlineRecognizer::create(&native_config).ok_or_else(|| {
            SpeechError::InvalidModel(format!(
                "sherpa-onnx could not load the streaming model in {} with provider {}",
                model.root.display(),
                config.inference.provider
            ))
        })?;
        let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
        caps.reports_partials = true;
        caps.reports_activity = true;
        caps.accepts_hints = accepts_hints;
        Ok(Self {
            recognizer: Arc::new(recognizer),
            caps,
            hotwords,
            endpoint_silence: config.endpoint_silence,
        })
    }
}

impl AsrBackend for SherpaStreaming {
    fn name(&self) -> &'static str {
        "sherpa-streaming"
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn open(
        &self,
        opts: &AsrOptions,
        events: AsrEvents,
    ) -> Result<Box<dyn AsrStream>, SpeechError> {
        let native = match self.hotwords.for_session(opts)? {
            Some(words) => self.recognizer.create_stream_with_hotwords(&words),
            None => self.recognizer.create_stream(),
        };
        Ok(Box::new(Stream {
            events,
            native,
            recognizer: self.recognizer.clone(),
            endpoint_silence: self.endpoint_silence,
            utterance: 0,
            last: String::new(),
            fed: 0,
            start: Duration::ZERO,
            tokens: None,
            speech: None,
            known: Duration::ZERO,
        }))
    }
}

impl std::fmt::Debug for SherpaStreaming {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SherpaStreaming")
            .field("caps", &self.caps)
            .finish_non_exhaustive()
    }
}

struct Stream {
    events: AsrEvents,
    recognizer: Arc<OnlineRecognizer>,
    native: OnlineStream,
    endpoint_silence: Duration,
    utterance: u64,
    /// The last partial text reported for the current utterance.
    last: String,
    /// Samples fed so far, excluding padding.
    fed: u64,
    /// Where the current utterance started: the previous endpoint.
    start: Duration,
    /// The times of the current utterance's first and last tokens.
    tokens: Option<(Duration, Duration)>,
    /// Where the speech in progress started, once `SpeechStarted` was
    /// sent for it and until it ends. A cut does not end it.
    speech: Option<Duration>,
    /// The last `ActivityKnown` sent.
    known: Duration,
}

fn at(samples: u64) -> Duration {
    SampleRate::HZ_16000.duration_of(samples)
}

/// A time the recognizer reports, in seconds.
fn seconds(value: f32) -> Duration {
    Duration::try_from_secs_f32(value.max(0.0)).unwrap_or(Duration::ZERO)
}

impl Stream {
    fn decode(&mut self, end_of_input: bool) -> Result<Vec<AsrEvent>, SpeechError> {
        while self.recognizer.is_ready(&self.native) {
            self.recognizer.decode(&self.native);
        }
        let result = self
            .recognizer
            .get_result(&self.native)
            .ok_or_else(|| native("no streaming result"))?;
        let mut events = Vec::new();
        let text = result.text.trim().to_owned();
        let fed = at(self.fed);
        let base = result.start_time.unwrap_or(0.0);
        let tokens = match &result.timestamps {
            Some(timestamps) if !timestamps.is_empty() => timestamps
                .first()
                .zip(timestamps.last())
                .map(|(first, last)| (seconds(base + first), seconds(base + last))),
            // Text without timestamps: date it as late as it can be.
            _ if !text.is_empty() => {
                let late = fed.saturating_sub(LATENCY).max(self.start);
                Some((late, late))
            }
            _ => None,
        };
        if let Some((first, last)) = tokens {
            let first = self.tokens.map_or(first, |(known, _)| known);
            self.tokens = Some((first, last.max(first)));
            if self.speech.is_none() {
                self.speech = Some(first);
                events.push(AsrEvent::SpeechStarted { at: first });
            }
        }
        let id = UtteranceId(self.utterance);
        if end_of_input || self.recognizer.is_endpoint(&self.native) {
            let committed = !text.is_empty() || !self.last.is_empty() || self.tokens.is_some();
            if committed {
                let (start, end) = self.tokens.unwrap_or((self.start, fed));
                // Rule 3 fires while speech goes on: a cut.
                let cut = !end_of_input && fed.saturating_sub(end) < self.endpoint_silence;
                if !cut && self.speech.take().is_some() {
                    events.push(AsrEvent::SpeechEnded {
                        at: end,
                        utterance: id,
                    });
                }
                events.push(AsrEvent::Segment(Segment {
                    utterance: id,
                    text,
                    start,
                    end,
                }));
                self.utterance += 1;
            }
            self.last.clear();
            self.tokens = None;
            self.start = fed;
            if !end_of_input {
                self.recognizer.reset(&self.native);
            }
        } else if text != self.last {
            self.last.clone_from(&text);
            events.push(AsrEvent::Partial(Partial {
                utterance: id,
                text,
            }));
        }
        let known = if end_of_input {
            fed
        } else {
            fed.saturating_sub(LATENCY)
        };
        let through = self.speech.map_or(known, |start| start.min(known));
        if through > self.known {
            self.known = through;
            events.push(AsrEvent::ActivityKnown { through });
        }
        Ok(events)
    }
}

impl AsrStream for Stream {
    fn accept(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        self.native.accept_waveform(RATE, samples);
        self.fed += samples.len() as u64;
        let events = self.decode(false)?;
        self.events.send_all(events);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SpeechError> {
        if self.fed == 0 {
            return Ok(());
        }
        self.native.accept_waveform(RATE, &vec![0.0; TAIL_PADDING]);
        self.native.input_finished();
        let events = self.decode(true)?;
        self.events.send_all(events);
        Ok(())
    }
}
