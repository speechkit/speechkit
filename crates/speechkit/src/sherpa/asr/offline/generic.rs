//! Offline families other than SenseVoice: transducers, Paraformer,
//! FireRed (AED and CTC), Qwen3-ASR, and FunASR-Nano.

use std::{fmt, sync::Arc};

use crate::{
    SampleRate, SpeechError,
    asr::{AsrCapabilities, AsrOptions},
    vad::OfflineRecognizer,
};
use sherpa_onnx::{OfflineRecognizer as NativeRecognizer, OfflineRecognizerConfig};

use crate::sherpa::{
    asr::{Hotwords, PreparedBias},
    bias,
    config::{AsrFamily, Inference, ModelFiles},
    layout::AsrFiles,
    load_failed,
};

/// Settings for [`GenericOffline`], checked by `AsrConfig`.
#[derive(Debug)]
pub(crate) struct OfflineConfig {
    /// The model's files.
    pub files: ModelFiles,
    /// The family. SenseVoice has its own type, [`SenseVoice`](super::SenseVoice).
    pub family: AsrFamily,
    /// Where and how the model runs.
    pub inference: Inference,
    /// Decoding bias; offline transducers only.
    pub bias: Option<PreparedBias>,
    /// Prompt phrases for every session; Qwen3-ASR and FunASR-Nano only.
    /// Empty means none.
    pub prompt_hints: Vec<String>,
}

/// What each offline family accepts, checked again when it loads.
///
/// # Errors
///
/// [`SpeechError::Unsupported`] for a family used here that does not fit,
/// or an option the family does not support.
pub(super) fn check_family_options(
    family: AsrFamily,
    bias: bool,
    prompt_hints: bool,
) -> Result<(), SpeechError> {
    let unsupported = |what: &str| {
        Err(SpeechError::Unsupported(format!(
            "the {family} family does not support {what}"
        )))
    };
    match family {
        AsrFamily::StreamingTransducer | AsrFamily::SenseVoice => Err(SpeechError::Unsupported(
            format!("the {family} family has its own backend type"),
        )),
        AsrFamily::OfflineTransducer if prompt_hints => unsupported("prompt hints"),
        AsrFamily::Qwen3Asr | AsrFamily::FunAsrNano if bias => unsupported("decoding bias"),
        AsrFamily::Paraformer | AsrFamily::FireRedCtc | AsrFamily::FireRedAed
            if bias || prompt_hints =>
        {
            unsupported("hints or decoding bias")
        }
        _ => Ok(()),
    }
}

/// An offline recognizer for one of the families [`OfflineConfig`] names.
///
/// None of these families accept a language override. Transducers with
/// bias accept session hints; Qwen3-ASR and FunASR-Nano take prompt hints
/// only when loaded.
pub(crate) struct GenericOffline {
    recognizer: Arc<NativeRecognizer>,
    family: AsrFamily,
    caps: AsrCapabilities,
    hotwords: Hotwords,
}

impl GenericOffline {
    /// Loads the model from the files and bias `AsrConfig` checked.
    ///
    /// # Errors
    ///
    /// [`SpeechError::Unsupported`] for options the family lacks,
    /// [`SpeechError::InvalidInput`] for bad settings or hints, and
    /// [`SpeechError::InvalidModel`] for a model the native library
    /// rejects.
    pub(crate) fn load(mut config: OfflineConfig) -> Result<Self, SpeechError> {
        let family = config.family;
        check_family_options(
            family,
            config.bias.is_some(),
            !config.prompt_hints.is_empty(),
        )?;
        config.inference.validate()?;
        let mut native = OfflineRecognizerConfig::default();
        native.model_config.num_threads = config.inference.threads_i32();
        native.model_config.provider = Some(config.inference.provider.as_str().into());
        let accepts_hints = config.bias.is_some();
        let bias = config.bias.take();
        let hotwords = configure(&mut native, &config.files, &config, bias)?;
        let recognizer = NativeRecognizer::create(&native)
            .ok_or_else(|| load_failed(family, &config.files.root))?;
        let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
        caps.accepts_hints = accepts_hints;
        caps.punctuated = matches!(family, AsrFamily::Qwen3Asr | AsrFamily::FunAsrNano);
        Ok(Self {
            recognizer: Arc::new(recognizer),
            family,
            caps,
            hotwords,
        })
    }
}

/// Points the native configuration at the family's files, and applies the
/// prepared bias or the prompt hints.
fn configure(
    native: &mut OfflineRecognizerConfig,
    model: &ModelFiles,
    config: &OfflineConfig,
    bias: Option<PreparedBias>,
) -> Result<Hotwords, SpeechError> {
    let path = |name: &String| Some(model.path(name).to_string_lossy().into_owned());
    let mc = &mut native.model_config;
    match &model.files {
        AsrFiles::Transducer {
            encoder,
            decoder,
            joiner,
            tokens,
            ..
        } => {
            mc.transducer.encoder = path(encoder);
            mc.transducer.decoder = path(decoder);
            mc.transducer.joiner = path(joiner);
            mc.tokens = path(tokens);
            if let Some(prepared) = bias {
                mc.modeling_unit = Some(prepared.unit.as_str().into());
                mc.bpe_vocab = prepared.bpe_vocab;
                native.decoding_method = Some("modified_beam_search".into());
                native.max_active_paths = 4;
                native.hotwords_score = prepared.score;
                return Ok(prepared.hotwords);
            }
        }
        AsrFiles::Flat {
            model: file,
            tokens,
        } => {
            mc.tokens = path(tokens);
            if config.family == AsrFamily::Paraformer {
                mc.paraformer.model = path(file);
            } else {
                mc.fire_red_asr_ctc.model = path(file);
            }
        }
        AsrFiles::FireRedAed {
            encoder,
            decoder,
            tokens,
        } => {
            mc.fire_red_asr.encoder = path(encoder);
            mc.fire_red_asr.decoder = path(decoder);
            mc.tokens = path(tokens);
        }
        AsrFiles::Qwen3Asr { .. } | AsrFiles::FunAsrNano { .. } => {
            configure_llm(native, model, config)?;
        }
    }
    Ok(Hotwords::default())
}

/// Qwen3-ASR and FunASR-Nano: a tokenizer directory and prompt hints.
fn configure_llm(
    native: &mut OfflineRecognizerConfig,
    model: &ModelFiles,
    config: &OfflineConfig,
) -> Result<(), SpeechError> {
    let path = |name: &String| Some(model.path(name).to_string_lossy().into_owned());
    let prompt = |funasr| bias::prompt(&config.prompt_hints, funasr);
    match &model.files {
        AsrFiles::Qwen3Asr {
            conv_frontend,
            encoder,
            decoder,
            tokenizer,
        } => {
            let qwen = &mut native.model_config.qwen3_asr;
            qwen.conv_frontend = path(conv_frontend);
            qwen.encoder = path(encoder);
            qwen.decoder = path(decoder);
            qwen.tokenizer = path(tokenizer);
            qwen.hotwords = prompt(false)?;
        }
        AsrFiles::FunAsrNano {
            encoder_adaptor,
            llm,
            embedding,
            tokenizer,
        } => {
            let nano = &mut native.model_config.funasr_nano;
            nano.encoder_adaptor = path(encoder_adaptor);
            nano.llm = path(llm);
            nano.embedding = path(embedding);
            nano.tokenizer = path(tokenizer);
            // The crate's defaults sample randomly with ITN off; use the
            // upstream C++ defaults: greedy decoding and ITN on.
            nano.temperature = 1e-6;
            nano.top_p = 0.8;
            nano.seed = 42;
            nano.max_new_tokens = 512;
            nano.itn = 1;
            nano.hotwords = prompt(true)?;
        }
        _ => {}
    }
    Ok(())
}

impl OfflineRecognizer for GenericOffline {
    fn name(&self) -> &'static str {
        "sherpa-offline"
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn recognize(&self, samples: &[f32], opts: &AsrOptions) -> Result<String, SpeechError> {
        let stream = match self.hotwords.for_session(opts)? {
            Some(words) => self.recognizer.create_stream_with_hotwords(&words),
            None => self.recognizer.create_stream(),
        };
        super::transcribe(&self.recognizer, &stream, samples)
    }
}

impl fmt::Debug for GenericOffline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GenericOffline")
            .field("family", &self.family)
            .field("caps", &self.caps)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Which options each family accepts or rejects, as a table.
    #[test]
    fn family_option_table() {
        use AsrFamily as F;
        let rows = [
            (F::OfflineTransducer, false, false, true),
            (F::OfflineTransducer, true, false, true),
            (F::OfflineTransducer, false, true, false),
            (F::Paraformer, false, false, true),
            (F::Paraformer, true, false, false),
            (F::Paraformer, false, true, false),
            (F::FireRedCtc, true, false, false),
            (F::FireRedCtc, false, true, false),
            (F::FireRedAed, false, false, true),
            (F::FireRedAed, true, false, false),
            (F::FireRedAed, false, true, false),
            (F::Qwen3Asr, false, true, true),
            (F::Qwen3Asr, true, false, false),
            (F::FunAsrNano, false, true, true),
            (F::FunAsrNano, true, false, false),
            (F::SenseVoice, false, false, false),
            (F::StreamingTransducer, false, false, false),
        ];
        for (family, bias, prompt, ok) in rows {
            let result = check_family_options(family, bias, prompt);
            assert_eq!(result.is_ok(), ok, "{family} bias={bias} prompt={prompt}");
            if let Err(error) = result {
                assert!(matches!(error, SpeechError::Unsupported(_)), "{error}");
            }
        }
    }
}
