//! Performance baselines. Run with
//! `cargo bench -p speechkit --all-features`.
#![expect(clippy::unwrap_used, reason = "benches fail loudly")]
#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use speechkit::{
    AudioBuffer, SampleRate,
    asr::{AsrEngine, AsrOptions},
    audio::{self, DecodeLimits, encode_wav},
    bench::{Chunker, Resampler},
};
use speechkit_testkit::asr::FakeAsr;

fn tone(rate: SampleRate, seconds: u32) -> Vec<f32> {
    let frames = rate.hz() * seconds;
    (0..frames)
        .map(|i| {
            #[expect(clippy::cast_precision_loss, reason = "small indices")]
            let t = i as f32 / rate.hz() as f32;
            (t * 440.0 * std::f32::consts::TAU).sin() * 0.3
        })
        .collect()
}

fn resampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("resample_1s");
    for (from, to) in [
        (SampleRate::HZ_48000, SampleRate::HZ_16000),
        (SampleRate::HZ_44100, SampleRate::HZ_16000),
        (SampleRate::HZ_16000, SampleRate::HZ_24000),
    ] {
        let input = tone(from, 1);
        group.throughput(Throughput::Elements(u64::from(from.hz())));
        group.bench_function(BenchmarkId::from_parameter(format!("{from}->{to}")), |b| {
            b.iter(|| {
                let mut resampler = Resampler::new(from, to).unwrap();
                let mut out = Vec::with_capacity(input.len());
                for chunk in input.chunks(1_600) {
                    resampler.process(black_box(chunk), &mut out).unwrap();
                }
                resampler.flush(&mut out).unwrap();
                out
            });
        });
    }
    group.finish();
}

/// The session's own cost per 100 ms chunk: queueing, the worker thread,
/// and result bookkeeping, with a backend that does no work.
fn session_overhead(c: &mut Criterion) {
    const CHUNKS: u32 = 100;
    let engine = AsrEngine::new(FakeAsr::new(speechkit_testkit::asr::Script::new()));
    let chunk = tone(SampleRate::HZ_16000, 1)[..1_600].to_vec();
    c.bench_function("asr_session_per_100ms_chunk", |b| {
        b.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                let started = Instant::now();
                let session = engine
                    .start(
                        SampleRate::HZ_16000,
                        AsrOptions::default(),
                        Duration::from_secs(10),
                    )
                    .unwrap();
                let deadline = Instant::now() + Duration::from_secs(10);
                for _ in 0..CHUNKS {
                    session.push(chunk.clone(), deadline).unwrap();
                }
                let _ = black_box(session.finish(deadline));
                total += started.elapsed() / CHUNKS;
            }
            total
        });
    });
}

fn decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    let wav = encode_wav(&AudioBuffer::new(
        SampleRate::HZ_16000,
        tone(SampleRate::HZ_16000, 10),
    ))
    .unwrap();
    group.throughput(Throughput::Bytes(wav.len() as u64));
    group.bench_function("wav_pcm16_16k_10s", |b| {
        b.iter(|| audio::decode(black_box(&wav), DecodeLimits::default()).unwrap());
    });
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/audio");
    for name in ["speech.flac", "speech.mp3"] {
        let Ok(bytes) = std::fs::read(fixtures.join(name)) else {
            continue;
        };
        group.throughput(Throughput::Bytes(bytes.len() as u64));
        group.bench_function(name, |b| {
            b.iter(|| audio::decode(black_box(&bytes), DecodeLimits::default()).unwrap());
        });
    }
    group.finish();
}

fn chunker(c: &mut Criterion) {
    let text = "Mr. Smith paid $3.14 for the tea, e.g. green or black... 今天天气很好，我们去公园吧！Then we went home. "
        .repeat(100);
    let mut group = c.benchmark_group("tts_chunker");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("mixed_text", |b| {
        b.iter(|| {
            let mut chunker = Chunker::new(300);
            let mut chunks = chunker.push(black_box(&text));
            chunks.extend(chunker.flush());
            chunks
        });
    });
    group.finish();
}

criterion_group!(benches, resampler, session_overhead, decoding, chunker);
criterion_main!(benches);
