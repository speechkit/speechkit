//! `speechkit speak`, `speechkit voices`, and speech in `speechkit serve`,
//! run in-process with a fake synthesis backend.
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{
    io::{Read, Write},
    net::TcpStream,
    process::ExitCode,
};

use speechkit::{
    SampleRate, SpeechError,
    asr::AsrEngine,
    audio::{self, DecodeLimits},
    tts::TtsEngine,
};
use speechkit_cli::{BackendArgs, BackendFactory, ServeArgs, TtsBackendArgs, run, serve};
use speechkit_testkit::{
    asr::FakeAsr,
    tts::{FakeTts, SAMPLES_PER_CHAR},
};

/// Fakes for both directions.
struct Fakes;

impl BackendFactory for Fakes {
    fn asr(&self, _: &BackendArgs) -> Result<AsrEngine, SpeechError> {
        Ok(AsrEngine::new(FakeAsr::hello_world()))
    }

    fn tts(&self, _: &TtsBackendArgs) -> Result<TtsEngine, SpeechError> {
        Ok(TtsEngine::new(FakeTts::plain()))
    }
}

/// A factory with recognition only, like one written before synthesis.
struct AsrOnly;

impl BackendFactory for AsrOnly {
    fn asr(&self, args: &BackendArgs) -> Result<AsrEngine, SpeechError> {
        Fakes.asr(args)
    }
}

fn call(args: &[&str], backends: &dyn BackendFactory) -> (ExitCode, String, String) {
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut command_line = vec!["speechkit"];
    command_line.extend_from_slice(args);
    let code = run(command_line, backends, &mut stdout, &mut stderr);
    (
        code,
        String::from_utf8(stdout).unwrap(),
        String::from_utf8(stderr).unwrap(),
    )
}

/// The largest `--timeout` means no limit, and must not overflow the clock.
#[test]
fn speak_with_the_longest_timeout() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("a.wav");
    let forever = u64::MAX.to_string();
    let (code, _, stderr) = call(
        &[
            "speak",
            "Hello there.",
            "--timeout",
            &forever,
            "--out",
            out.to_str().unwrap(),
        ],
        &Fakes,
    );
    assert_eq!(code, ExitCode::SUCCESS, "{stderr}");
    assert!(out.exists());
}

#[test]
fn speak_writes_a_wav_file() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("a.wav");
    let (code, stdout, stderr) = call(
        &[
            "speak",
            "Hello there.",
            "--voice",
            "alpha",
            "--out",
            out.to_str().unwrap(),
        ],
        &Fakes,
    );
    assert_eq!(code, ExitCode::SUCCESS, "{stderr}");
    assert!(stdout.is_empty());
    assert!(stderr.contains("wrote"), "{stderr}");
    let audio = audio::read(&out, DecodeLimits::default()).unwrap();
    assert_eq!(audio.sample_rate, SampleRate::HZ_16000);
    assert_eq!(audio.samples.len(), "Hello there.".len() * SAMPLES_PER_CHAR);
}

#[test]
fn speak_reads_a_text_file() {
    let dir = tempfile::tempdir().unwrap();
    let text = dir.path().join("in.txt");
    std::fs::write(&text, "第一句。第二句。").unwrap();
    let out = dir.path().join("a.wav");
    let (code, _, stderr) = call(
        &[
            "speak",
            "--file",
            text.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ],
        &Fakes,
    );
    assert_eq!(code, ExitCode::SUCCESS, "{stderr}");
    assert_eq!(
        audio::read(&out, DecodeLimits::default())
            .unwrap()
            .samples
            .len(),
        8 * SAMPLES_PER_CHAR
    );
}

#[test]
fn speak_usage_errors() {
    // No text, no output, or both texts.
    assert_eq!(
        call(&["speak", "--out", "a.wav"], &Fakes).0,
        ExitCode::from(2)
    );
    assert_eq!(call(&["speak", "hi"], &Fakes).0, ExitCode::from(2));
    assert_eq!(
        call(
            &["speak", "hi", "--file", "x.txt", "--out", "a.wav"],
            &Fakes
        )
        .0,
        ExitCode::from(2)
    );
}

#[test]
fn speak_rejects_bad_options_with_exit_3() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("a.wav");
    let out = out.to_str().unwrap();
    for extra in [["--voice", "robot"], ["--speed", "9"]] {
        let mut args = vec!["speak", "hi", "--out", out];
        args.extend(extra);
        let (code, _, stderr) = call(&args, &Fakes);
        assert_eq!(code, ExitCode::from(3), "{extra:?}: {stderr}");
        assert!(stderr.starts_with("error: invalid input"), "{stderr}");
    }
    let (code, _, stderr) = call(&["speak", "hi", "--out", out], &AsrOnly);
    assert_eq!(code, ExitCode::from(3));
    assert!(stderr.contains("no speech synthesis"), "{stderr}");
}

#[test]
fn voices_lists_ids_and_languages() {
    let (code, stdout, stderr) = call(&["voices"], &Fakes);
    assert_eq!(code, ExitCode::SUCCESS, "{stderr}");
    assert_eq!(stdout, "alpha\ten\nbeta\tzh\n");
}

fn serve_args(extra: &[&str]) -> ServeArgs {
    use clap::Parser;
    #[derive(Parser)]
    struct Wrapper {
        #[command(flatten)]
        serve: ServeArgs,
    }
    let mut argv = vec!["x", "--bind", "127.0.0.1:0"];
    argv.extend_from_slice(extra);
    Wrapper::parse_from(argv).serve
}

fn request(address: std::net::SocketAddr, raw: &str) -> Vec<u8> {
    let mut stream = TcpStream::connect(address).unwrap();
    stream.write_all(raw.as_bytes()).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    response
}

#[test]
fn serve_with_a_tts_model_answers_speech() {
    let mut stderr = Vec::new();
    let handle = serve(
        &serve_args(&["--tts-model", "/models/piper-amy"]),
        &Fakes,
        &mut stderr,
    )
    .unwrap();
    let body = r#"{"input":"Hi.","response_format":"pcm"}"#;
    let response = request(
        handle.local_addr(),
        &format!(
            "POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    );
    let text = String::from_utf8_lossy(&response);
    assert!(text.starts_with("HTTP/1.1 200"), "{text}");
    assert!(text.contains("audio/pcm"), "{text}");
    let models = request(
        handle.local_addr(),
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    let models = String::from_utf8(models).unwrap();
    assert!(models.contains("\"id\":\"piper-amy\""), "{models}");
    handle.shutdown().unwrap();
}

#[test]
fn serve_without_tts_flags_has_no_speech() {
    let mut stderr = Vec::new();
    let handle = serve(&serve_args(&[]), &AsrOnly, &mut stderr).unwrap();
    let body = r#"{"input":"Hi."}"#;
    let response = request(
        handle.local_addr(),
        &format!(
            "POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    );
    assert!(String::from_utf8_lossy(&response).starts_with("HTTP/1.1 404"));
    handle.shutdown().unwrap();
}

#[test]
fn speak_device_needs_play() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("a.wav");
    let (code, _, stderr) = call(
        &[
            "speak",
            "Hi.",
            "--out",
            out.to_str().unwrap(),
            "--device",
            "x",
        ],
        &Fakes,
    );
    assert_eq!(code, ExitCode::from(speechkit_cli::EXIT_USAGE), "{stderr}");
    assert!(stderr.contains("--device needs --play"), "{stderr}");
    assert!(!out.exists());
}

#[test]
fn an_empty_device_name_is_a_usage_error() {
    for args in [
        &["speak", "Hi.", "--play", "--device", ""][..],
        &["speak", "Hi.", "--play", "--device", "  "][..],
        &["mic", "--device", ""][..],
    ] {
        let (code, _, stderr) = call(args, &Fakes);
        assert_eq!(code, ExitCode::from(speechkit_cli::EXIT_USAGE), "{stderr}");
        assert!(stderr.contains("the device name is empty"), "{stderr}");
    }
}
