//! The CLI, run in-process with fake backends.
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{
    io::{Read, Write},
    net::TcpStream,
    path::PathBuf,
    process::ExitCode,
    time::Duration,
};

use speechkit::{
    AudioBuffer, SampleRate, SpeechError,
    asr::{AsrEngine, AsrLimits},
    audio::encode_wav,
};
use speechkit_cli::{BackendArgs, BackendFactory, ServeArgs, run, serve};
use speechkit_testkit::{
    Gate,
    asr::{FakeAsr, Script, Step, Trigger},
    contract::asr::tone,
};

/// A factory that always builds an engine on the given fake.
struct Fake(FakeAsr);

impl BackendFactory for Fake {
    fn asr(&self, _: &BackendArgs) -> Result<AsrEngine, SpeechError> {
        Ok(AsrEngine::new(self.0.clone()))
    }
}

/// A factory whose sessions hold only 50 ms, so longer chunks are refused.
struct SmallChunks(FakeAsr);

impl BackendFactory for SmallChunks {
    fn asr(&self, _: &BackendArgs) -> Result<AsrEngine, SpeechError> {
        Ok(AsrEngine::new(self.0.clone())
            .with_limits(AsrLimits::default().with_input_queue(Duration::from_millis(50))))
    }
}

/// A factory whose backend cannot be built.
struct Broken;

impl BackendFactory for Broken {
    fn asr(&self, _: &BackendArgs) -> Result<AsrEngine, SpeechError> {
        Err(SpeechError::backend("broken", true, "service unavailable"))
    }
}

fn wav_file(seconds: usize) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("speech.wav");
    let audio = AudioBuffer::new(SampleRate::HZ_16000, tone(16_000 * seconds));
    std::fs::write(&path, encode_wav(&audio).unwrap()).unwrap();
    (dir, path)
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

#[test]
fn transcribe_writes_only_the_transcript_to_stdout() {
    let (_dir, file) = wav_file(1);
    let (code, stdout, stderr) = call(
        &["transcribe", file.to_str().unwrap()],
        &Fake(FakeAsr::hello_world()),
    );
    assert_eq!(code, ExitCode::SUCCESS, "{stderr}");
    assert_eq!(stdout, "hello world\n");
    assert!(stderr.contains("] hello"), "{stderr}");
    assert!(stderr.contains("] world"), "{stderr}");
}

/// `--timeout` is a number of seconds, and the largest one means no limit:
/// adding it to the current time must not panic.
#[test]
fn the_longest_timeout_means_no_limit() {
    let (_dir, file) = wav_file(1);
    let forever = u64::MAX.to_string();
    for command in ["transcribe", "stream"] {
        let (code, stdout, stderr) = call(
            &[command, file.to_str().unwrap(), "--timeout", &forever],
            &Fake(FakeAsr::hello_world()),
        );
        assert_eq!(code, ExitCode::SUCCESS, "{command}: {stderr}");
        assert_eq!(stdout, "hello world\n", "{command}");
    }
}

#[test]
fn transcribe_respects_a_small_max_chunk() {
    let (_dir, file) = wav_file(1);
    let (code, stdout, stderr) = call(
        &["transcribe", file.to_str().unwrap()],
        &SmallChunks(FakeAsr::hello_world()),
    );
    assert_eq!(code, ExitCode::SUCCESS, "{stderr}");
    assert_eq!(stdout, "hello world\n");
}

#[test]
fn transcribe_formats() {
    let (_dir, file) = wav_file(1);
    let path = file.to_str().unwrap();
    let (code, stdout, _) = call(
        &["transcribe", path, "--format", "json"],
        &Fake(FakeAsr::hello_world()),
    );
    assert_eq!(code, ExitCode::SUCCESS);
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["text"], "hello world");
    assert_eq!(value["segments"][0]["end_ms"], 500);
    let (code, stdout, _) = call(
        &["transcribe", path, "--format", "srt", "-q"],
        &Fake(FakeAsr::hello_world()),
    );
    assert_eq!(code, ExitCode::SUCCESS);
    assert!(
        stdout.starts_with("1\n00:00:00,000 --> 00:00:00,500\nhello\n\n2\n"),
        "{stdout}"
    );
}

#[test]
fn quiet_hides_progress() {
    let (_dir, file) = wav_file(1);
    let (code, stdout, stderr) = call(
        &["-q", "transcribe", file.to_str().unwrap()],
        &Fake(FakeAsr::hello_world()),
    );
    assert_eq!(code, ExitCode::SUCCESS);
    assert_eq!(stdout, "hello world\n");
    assert!(stderr.contains("] hello"), "segments still print: {stderr}");
}

#[test]
fn exit_codes() {
    let (_dir, file) = wav_file(1);
    let path = file.to_str().unwrap();
    let (code, _, stderr) = call(
        &["transcribe", "/nonexistent/a.wav"],
        &Fake(FakeAsr::hello_world()),
    );
    assert_eq!(code, ExitCode::from(3), "{stderr}");
    assert!(stderr.starts_with("error: "), "{stderr}");
    let (code, _, stderr) = call(&["transcribe", path], &Broken);
    assert_eq!(code, ExitCode::from(4), "{stderr}");
    assert!(stderr.contains("service unavailable"), "{stderr}");
    let failing = FakeAsr::new(Script::new().then(
        Trigger::AfterSamples(1),
        Step::Fail(SpeechError::backend("fake", false, "boom")),
    ));
    let (code, _, stderr) = call(&["transcribe", path], &Fake(failing));
    assert_eq!(code, ExitCode::from(4), "{stderr}");
    let (code, _, stderr) = call(
        &["transcribe", path, "--language", "zh"],
        &Fake(FakeAsr::hello_world()),
    );
    assert_eq!(code, ExitCode::from(3), "{stderr}");
    let (code, _, stderr) = call(&["transcribe"], &Fake(FakeAsr::hello_world()));
    assert_eq!(code, ExitCode::from(2), "{stderr}");
}

#[test]
fn stream_retries_on_a_full_queue() {
    let gate = Gate::new();
    let fake = FakeAsr::new(
        Script::new()
            .then(
                Trigger::AfterSamples(1),
                Step::BlockUntilReleased(gate.clone()),
            )
            .then(Trigger::OnFinish, Step::Segment(0, "streamed")),
    );
    let releaser = std::thread::spawn({
        let gate = gate.clone();
        move || {
            assert!(gate.wait_entered(1, Duration::from_secs(10)));
            std::thread::sleep(Duration::from_millis(200));
            gate.release();
        }
    });
    let (_dir, file) = wav_file(3);
    let (code, stdout, stderr) = call(&["stream", file.to_str().unwrap()], &Fake(fake));
    releaser.join().unwrap();
    assert_eq!(code, ExitCode::SUCCESS, "{stderr}");
    assert_eq!(stdout, "streamed\n");
    assert!(stderr.contains("the queue was full"), "{stderr}");
    let (code, _, _) = call(
        &["stream", file.to_str().unwrap(), "--chunk-ms", "0"],
        &Fake(FakeAsr::hello_world()),
    );
    assert_eq!(code, ExitCode::from(2));
}

fn serve_args(bind: &str, auth: Option<&str>) -> ServeArgs {
    use clap::Parser;
    #[derive(Parser)]
    struct Wrapper {
        #[command(flatten)]
        serve: ServeArgs,
    }
    let mut argv = vec!["x", "--bind", bind];
    if let Some(var) = auth {
        argv.extend(["--auth-token-env", var]);
    }
    Wrapper::parse_from(argv).serve
}

fn get(address: std::net::SocketAddr, path: &str) -> String {
    let mut stream = TcpStream::connect(address).unwrap();
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    response
}

#[test]
fn serve_answers_health_on_port_zero() {
    let mut stderr = Vec::new();
    let handle = serve(
        &serve_args("127.0.0.1:0", None),
        &Fake(FakeAsr::hello_world()),
        &mut stderr,
    )
    .unwrap();
    let response = get(handle.local_addr(), "/health");
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("\"active_sessions\":0"), "{response}");
    handle.shutdown().unwrap();
    let stderr = String::from_utf8(stderr).unwrap();
    assert!(
        stderr.contains("listening on http://127.0.0.1:"),
        "{stderr}"
    );
    assert!(!stderr.contains("warning"), "{stderr}");
}

#[test]
fn serve_warns_without_auth_off_loopback() {
    let mut stderr = Vec::new();
    let handle = serve(
        &serve_args("0.0.0.0:0", None),
        &Fake(FakeAsr::hello_world()),
        &mut stderr,
    )
    .unwrap();
    handle.shutdown().unwrap();
    let stderr = String::from_utf8(stderr).unwrap();
    assert!(
        stderr.contains("warning: serving on 0.0.0.0:0 without authentication"),
        "{stderr}"
    );
}

#[test]
fn serve_needs_the_token_variable() {
    let mut stderr = Vec::new();
    let result = serve(
        &serve_args("0.0.0.0:0", Some("SPEECHKIT_TEST_UNSET_TOKEN_VARIABLE")),
        &Fake(FakeAsr::hello_world()),
        &mut stderr,
    );
    assert_eq!(result.err().map(|e| e.code), Some(2));
}
