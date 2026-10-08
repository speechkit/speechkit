//! Round trips: synthesize fixed sentences with each local
//! TTS model, transcribe them with local ASR, and check the character
//! error rate against a threshold stored per pair. This catches broken
//! voices, wrong sample rates, and clipping.
//!
//! Ignored by default; it runs nightly in `models.yml` once the models
//! listed as pending in `fixtures/manifest.json` are fetched. The
//! thresholds are provisional until the first recorded run.
#![cfg(feature = "sherpa")]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use speechkit::sherpa::{AsrConfig, TtsConfig};
use speechkit::{
    SampleRate,
    asr::{AsrEngine, AsrOptions},
    tts::{TtsEngine, TtsOptions},
};
use speechkit_testkit::{gates::model_dir, metrics::cer, secs};

const ENGLISH: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "Please call me back after the meeting.",
    "It is raining in the city today.",
    "She bought three apples and a loaf of bread.",
    "The train leaves at seven in the morning.",
    "Turn left at the next traffic light.",
    "We are going to the library this afternoon.",
    "My favorite color is green.",
];

const CHINESE: &[&str] = &[
    "今天天气很好，我们一起去公园散步吧。",
    "请把窗户关上，外面有点冷。",
    "他每天早上七点起床。",
    "这家餐厅的菜非常好吃。",
    "我明天要去北京出差。",
    "图书馆周末也开门。",
    "火车马上就要开了。",
    "你能帮我拿一下那本书吗？",
];

const MIXED: &[&str] = &[
    "我今天用 iPhone 拍了很多照片。",
    "明天的 meeting 改到下午三点。",
    "请把这个 email 转发给我。",
    "这个 app 很好用。",
];

/// A TTS model, the voice it reads with, the ASR model that checks it,
/// the sentences it reads, and the largest mean CER accepted.
struct Pair {
    tts: &'static str,
    /// The speaker ID, or `None` for the model's default.
    voice: Option<&'static str>,
    asr: Asr,
    sentences: &'static [&'static [&'static str]],
    max_cer: f64,
}

#[derive(Clone, Copy)]
enum Asr {
    StreamingEn,
    SenseVoice,
}

const PAIRS: &[Pair] = &[
    Pair {
        tts: "tts-piper-en",
        voice: None,
        asr: Asr::StreamingEn,
        sentences: &[ENGLISH],
        max_cer: 0.15,
    },
    Pair {
        tts: "tts-kokoro-en",
        voice: None,
        asr: Asr::SenseVoice,
        sentences: &[ENGLISH],
        max_cer: 0.15,
    },
    Pair {
        tts: "tts-matcha-zh",
        voice: None,
        asr: Asr::SenseVoice,
        sentences: &[CHINESE],
        max_cer: 0.15,
    },
    Pair {
        tts: "tts-kokoro-multi",
        // A Chinese voice (zf_xiaobei). The default, an English voice,
        // reads the Chinese in mixed sentences as pinyin; this one reads
        // Chinese, English, and mixed text correctly.
        voice: Some("45"),
        asr: Asr::SenseVoice,
        sentences: &[CHINESE, ENGLISH, MIXED],
        max_cer: 0.2,
    },
];

fn asr(kind: Asr) -> Option<AsrEngine> {
    let config = match kind {
        Asr::StreamingEn => AsrConfig::streaming(model_dir("streaming-en")?),
        Asr::SenseVoice => AsrConfig::offline(model_dir("sense-voice")?, model_dir("silero-vad")?),
    };
    Some(AsrEngine::new(config.load().unwrap()))
}

fn round_trip(pair: &Pair) {
    let (Some(dir), Some(asr)) = (model_dir(pair.tts), asr(pair.asr)) else {
        return;
    };
    let tts = TtsEngine::new(TtsConfig::new(dir).load().unwrap());
    let mut rates = Vec::new();
    for sentence in pair.sentences.iter().flat_map(|group| group.iter()) {
        let mut options = TtsOptions::default().with_sample_rate(SampleRate::HZ_16000);
        if let Some(voice) = pair.voice {
            options = options.with_voice(voice);
        }
        let audio = tts.synthesize(sentence, options, secs(120)).unwrap();
        let peak = audio.samples.iter().fold(0.0_f32, |m, s| m.max(s.abs()));
        assert!(peak < 0.999, "{}: {sentence:?} clips", pair.tts);
        let outcome = asr
            .transcribe(&audio, AsrOptions::default(), secs(120))
            .unwrap();
        let heard = outcome.text();
        let rate = cer(sentence, &heard);
        eprintln!(
            "{}: CER {rate:.3} for {sentence:?}, heard {heard:?}",
            pair.tts
        );
        rates.push(rate);
    }
    #[expect(clippy::cast_precision_loss, reason = "a handful of sentences")]
    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    assert!(
        mean <= pair.max_cer,
        "{}: mean CER {mean:.3} exceeds {}",
        pair.tts,
        pair.max_cer
    );
}

#[test]
#[ignore = "needs TTS and ASR models"]
fn round_trip_piper_en() {
    round_trip(&PAIRS[0]);
}

#[test]
#[ignore = "needs TTS and ASR models"]
fn round_trip_kokoro_en() {
    round_trip(&PAIRS[1]);
}

#[test]
#[ignore = "needs TTS and ASR models"]
fn round_trip_matcha_zh() {
    round_trip(&PAIRS[2]);
}

#[test]
#[ignore = "needs TTS and ASR models"]
fn round_trip_kokoro_multilingual() {
    round_trip(&PAIRS[3]);
}
