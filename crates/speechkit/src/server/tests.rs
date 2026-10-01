//! Slow clients are cut off, over a real socket. The header and body
//! timeouts have no public setters, so these tests set them here.

use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    time::{Duration, Instant},
};

use super::{RunningServer, Server};
use crate::{
    SampleRate, SpeechError,
    asr::{AsrBackend, AsrCapabilities, AsrEngine, AsrEvents, AsrOptions, AsrStream},
};

/// A recognizer that hears nothing.
struct Silent(AsrCapabilities);

impl AsrBackend for Silent {
    fn name(&self) -> &'static str {
        "silent"
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.0
    }

    fn open(&self, _: &AsrOptions, _: AsrEvents) -> Result<Box<dyn AsrStream>, SpeechError> {
        Ok(Box::new(Nothing))
    }
}

struct Nothing;

impl AsrStream for Nothing {
    fn accept(&mut self, _: &[f32]) -> Result<(), SpeechError> {
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SpeechError> {
        Ok(())
    }
}

/// Starts a server on a free port, with `configure` applied.
fn start(configure: impl FnOnce(&mut Server)) -> (tokio::runtime::Runtime, RunningServer) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let mut server = Server::new()
        .with_asr(
            AsrEngine::new(Silent(AsrCapabilities::new(SampleRate::HZ_16000))),
            "silent",
        )
        .with_bind("127.0.0.1:0");
    configure(&mut server);
    let running = runtime
        .block_on(server.start(std::future::pending()))
        .unwrap();
    (runtime, running)
}

/// Reads until the server closes the connection, returning what it sent
/// and how long that took.
fn read_to_close(mut stream: TcpStream) -> (String, Duration) {
    let started = Instant::now();
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let mut response = Vec::new();
    let _ = stream.read_to_end(&mut response);
    (
        String::from_utf8_lossy(&response).into_owned(),
        started.elapsed(),
    )
}

fn connect(address: SocketAddr) -> TcpStream {
    TcpStream::connect(address).unwrap()
}

#[test]
fn slow_headers_close_the_connection() {
    let (runtime, server) = start(|server| server.header_timeout = Duration::from_millis(300));
    let mut stream = connect(server.local_addr());
    // Half a request line, then nothing.
    stream.write_all(b"GET /health HT").unwrap();
    let (_, waited) = read_to_close(stream);
    assert!(waited < Duration::from_secs(5), "{waited:?}");
    runtime.block_on(server.shutdown()).unwrap();
}

#[test]
fn slow_bodies_are_cut_off() {
    let (runtime, server) = start(|server| server.body_timeout = Duration::from_millis(300));
    let mut stream = connect(server.local_addr());
    // Promise a body, send a little of it, then stall.
    stream
        .write_all(
            b"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\n\
              Content-Type: multipart/form-data; boundary=b\r\nContent-Length: 100000\r\n\r\n--b\r\n",
        )
        .unwrap();
    let (response, waited) = read_to_close(stream);
    assert!(waited < Duration::from_secs(5), "{waited:?}");
    assert!(
        response.is_empty() || !response.starts_with("HTTP/1.1 200"),
        "{response}"
    );
    runtime.block_on(server.shutdown()).unwrap();
}

#[test]
fn a_prompt_client_is_served() {
    let (runtime, server) = start(|server| server.header_timeout = Duration::from_millis(300));
    let mut stream = connect(server.local_addr());
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .unwrap();
    let (response, _) = read_to_close(stream);
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    runtime.block_on(server.shutdown()).unwrap();
}
