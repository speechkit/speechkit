//! Offline recognizers, which transcribe one utterance per call.

mod generic;
mod sense_voice;

pub(crate) use generic::{GenericOffline, OfflineConfig};
pub(crate) use sense_voice::{SenseVoice, SenseVoiceConfig, SenseVoiceLanguage};

/// Decodes `samples` on a fresh `stream` of `recognizer` and returns the
/// text.
fn transcribe(
    recognizer: &sherpa_onnx::OfflineRecognizer,
    stream: &sherpa_onnx::OfflineStream,
    samples: &[f32],
) -> Result<String, crate::SpeechError> {
    stream.accept_waveform(super::RATE, samples);
    recognizer.decode(stream);
    Ok(stream
        .get_result()
        .ok_or_else(|| super::native("no offline result"))?
        .text)
}
