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

/// A timeout says so: the session fails with `DeadlineExceeded`, not as if
/// the user had cancelled it.
#[test]
fn a_timeout_is_reported_as_one() {
    let (_dir, file) = wav_file(1);
    for command in ["transcribe", "stream"] {
        let gate = Gate::new();
        let stuck = FakeAsr::new(
            Script::new().then(Trigger::OnFinish, Step::BlockUntilReleased(gate.clone())),
        );
        let (code, stdout, stderr) = call(
            &[command, file.to_str().unwrap(), "--timeout", "1"],
            &Fake(stuck),
        );
        gate.release();
        assert_eq!(
            code,
            ExitCode::from(speechkit_cli::EXIT_BACKEND),
            "{stderr}"
        );
        assert_eq!(stdout, "", "{command}");
        assert!(
            stderr.contains("error: deadline exceeded"),
            "{command}: {stderr}"
        );
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

/// A host name binds as its addresses do, and `localhost` is local.
#[test]
fn serve_binds_a_host_name() {
    let mut stderr = Vec::new();
    let handle = serve(
        &serve_args("localhost:0", None),
        &Fake(FakeAsr::hello_world()),
        &mut stderr,
    )
    .unwrap();
    assert!(handle.local_addr().ip().is_loopback());
    let response = get(handle.local_addr(), "/health");
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    handle.shutdown().unwrap();
    let stderr = String::from_utf8(stderr).unwrap();
    assert!(!stderr.contains("warning"), "{stderr}");
}

/// A bind address that does not resolve is a usage error, found before
/// the backend is built: loading a model first would make the user wait
/// for nothing.
#[test]
fn serve_checks_the_bind_address_first() {
    let mut stderr = Vec::new();
    // No unknown host name: some resolvers, such as a proxy's, answer
    // every name, so only addresses that fail before any lookup are used.
    for bind in ["localhost", "127.0.0.1:port"] {
        let result = serve(&serve_args(bind, None), &Broken, &mut stderr);
        let error = result.err().unwrap();
        assert_eq!(error.code, 2, "{bind}: {}", error.message);
        assert!(error.message.contains(bind), "{}", error.message);
    }
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

/// Every failure path gives its exit code and a message that says what
/// went wrong in the user's terms, on stderr after `error: `, with nothing
/// on stdout. Items 5 and 6 of the review were both the CLI saying the
/// wrong thing on a failure path.
#[test]
#[expect(clippy::too_many_lines, reason = "a table with one row per failure")]
fn failures_give_their_code_and_message() {
    struct Case {
        name: &'static str,
        args: Vec<String>,
        backends: Box<dyn BackendFactory>,
        code: u8,
        says: &'static [&'static str],
        not: &'static [&'static str],
        /// Released after the call, for a backend that blocks.
        gate: Option<Gate>,
    }
    let (_dir, good) = wav_file(1);
    let good = good.to_str().unwrap().to_owned();
    let scratch = tempfile::tempdir().unwrap();
    let bad = scratch.path().join("bad.wav");
    std::fs::write(&bad, b"not audio at all").unwrap();
    let bad = bad.to_str().unwrap().to_owned();
    let empty = scratch.path().join("empty");
    std::fs::create_dir(&empty).unwrap();
    let empty = empty.to_str().unwrap().to_owned();
    let args = |list: &[&str]| list.iter().map(|&arg| arg.to_owned()).collect::<Vec<_>>();
    let stuck = || {
        let gate = Gate::new();
        let fake = FakeAsr::new(
            Script::new().then(Trigger::OnFinish, Step::BlockUntilReleased(gate.clone())),
        );
        (Box::new(Fake(fake)) as Box<dyn BackendFactory>, gate)
    };
    let (stuck_transcribe, transcribe_gate) = stuck();
    let (stuck_stream, stream_gate) = stuck();
    let cases = vec![
        Case {
            name: "a timeout",
            args: args(&["transcribe", &good, "--timeout", "1"]),
            backends: stuck_transcribe,
            code: speechkit_cli::EXIT_BACKEND,
            says: &["deadline exceeded"],
            not: &["cancel"],
            gate: Some(transcribe_gate),
        },
        Case {
            name: "a timeout while streaming",
            args: args(&["stream", &good, "--timeout", "1"]),
            backends: stuck_stream,
            code: speechkit_cli::EXIT_BACKEND,
            says: &["deadline exceeded"],
            not: &["cancel"],
            gate: Some(stream_gate),
        },
        Case {
            name: "a missing audio file",
            args: args(&["transcribe", "/nonexistent/a.wav"]),
            backends: Box::new(Fake(FakeAsr::hello_world())),
            code: speechkit_cli::EXIT_INPUT,
            // The OS's own wording follows, which differs on Windows.
            says: &["cannot decode audio", "/nonexistent/a.wav"],
            not: &[],
            gate: None,
        },
        Case {
            name: "a file that is not audio",
            args: args(&["transcribe", &bad]),
            backends: Box::new(Fake(FakeAsr::hello_world())),
            code: speechkit_cli::EXIT_INPUT,
            says: &["not a supported audio format", "supported formats: WAV"],
            not: &["probe", "format reader"],
            gate: None,
        },
        Case {
            name: "a missing model directory",
            args: args(&["transcribe", &good, "--model", "/nonexistent/model"]),
            backends: Box::new(speechkit_cli::Backends::default()),
            code: speechkit_cli::EXIT_INPUT,
            says: &["invalid model", "/nonexistent/model"],
            not: &[],
            gate: None,
        },
        Case {
            name: "a model directory without a model",
            args: args(&["transcribe", &good, "--model", &empty]),
            backends: Box::new(speechkit_cli::Backends::default()),
            code: speechkit_cli::EXIT_INPUT,
            says: &["invalid model", "encoder"],
            not: &[],
            gate: None,
        },
        Case {
            name: "no model at all",
            args: args(&["transcribe", &good]),
            backends: Box::new(speechkit_cli::Backends::default()),
            code: speechkit_cli::EXIT_INPUT,
            says: &["--model"],
            not: &[],
            gate: None,
        },
        Case {
            name: "a backend that cannot start",
            args: args(&["transcribe", &good]),
            backends: Box::new(Broken),
            code: speechkit_cli::EXIT_BACKEND,
            says: &["service unavailable"],
            not: &[],
            gate: None,
        },
        Case {
            name: "a bind address with a bad port",
            args: args(&["serve", "--bind", "127.0.0.1:port"]),
            backends: Box::new(Broken),
            code: speechkit_cli::EXIT_USAGE,
            says: &["127.0.0.1:port", "port"],
            not: &["service unavailable"],
            gate: None,
        },
        Case {
            name: "a bind address without a port",
            args: args(&["serve", "--bind", "localhost"]),
            backends: Box::new(Broken),
            code: speechkit_cli::EXIT_USAGE,
            says: &["localhost"],
            not: &["service unavailable"],
            gate: None,
        },
        Case {
            name: "a timeout that is not a number",
            args: args(&["transcribe", &good, "--timeout", "soon"]),
            backends: Box::new(Fake(FakeAsr::hello_world())),
            code: speechkit_cli::EXIT_USAGE,
            says: &["--timeout", "soon"],
            not: &[],
            gate: None,
        },
    ];
    for case in cases {
        let args: Vec<&str> = case.args.iter().map(String::as_str).collect();
        let (code, stdout, stderr) = call(&args, &*case.backends);
        if let Some(gate) = &case.gate {
            gate.release();
        }
        let name = case.name;
        assert_eq!(code, ExitCode::from(case.code), "{name}: {stderr}");
        assert_eq!(stdout, "", "{name}: stdout");
        assert!(stderr.starts_with("error: "), "{name}: {stderr}");
        for wanted in case.says {
            assert!(stderr.contains(wanted), "{name}: no {wanted:?} in {stderr}");
        }
        for unwanted in case.not {
            assert!(
                !stderr.contains(unwanted),
                "{name}: {unwanted:?} in {stderr}"
            );
        }
    }
}

/// `serve --check` builds the backends, which loads their models, and
/// binds the address, then exits without serving: 0 when both worked, and
/// the usual failure otherwise. A deployment can tell a bad flag or model
/// from a slow load before it takes traffic.
#[test]
fn serve_check_starts_and_exits() {
    let check = ["serve", "--check", "--bind", "127.0.0.1:0"];
    let (code, stdout, stderr) = call(&check, &Fake(FakeAsr::hello_world()));
    assert_eq!(code, ExitCode::SUCCESS, "{stderr}");
    assert_eq!(stdout, "");
    assert!(
        stderr.contains("listening on http://127.0.0.1:"),
        "{stderr}"
    );
    assert!(stderr.contains("check passed"), "{stderr}");

    let (code, _, stderr) = call(&check, &Broken);
    assert_eq!(
        code,
        ExitCode::from(speechkit_cli::EXIT_BACKEND),
        "{stderr}"
    );
    assert!(stderr.contains("service unavailable"), "{stderr}");

    // An address already in use.
    let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let bind = taken.local_addr().unwrap().to_string();
    let (code, _, stderr) = call(
        &["serve", "--check", "--bind", &bind],
        &Fake(FakeAsr::hello_world()),
    );
    assert_ne!(code, ExitCode::SUCCESS, "{stderr}");
    assert!(stderr.starts_with("error: "), "{stderr}");
    assert!(!stderr.contains("check passed"), "{stderr}");
}
