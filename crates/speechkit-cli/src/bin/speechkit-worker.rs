//! Runs a speechkit recognizer in its own process. Started by
//! `speechkit::sherpa::process::IsolatedAsr`.

fn main() -> std::process::ExitCode {
    speechkit::sherpa::process::worker_main()
}
