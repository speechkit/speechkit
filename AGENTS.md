# AGENTS.md

How to work in this repository, for people and AI coding agents.

## Project

speechkit is a Rust library and CLI for speech recognition (ASR) and speech synthesis (TTS).

The workspace has three crates:

- `crates/speechkit`: the library. The core (`speech`, `asr`, `tts`, `vad`, `wake`, `audio`) is always built; `sherpa`, `cloud`, `io` (feature `devices`), and `server` are modules behind features.
- `crates/speechkit-cli`: the `speechkit` command and the `speechkit-worker` crash-isolation binary.
- `crates/speechkit-testkit`: unpublished fakes and contract suites.

## Workflow

- Check third-party API details (sherpa-onnx, OpenAI, DashScope) against current upstream docs, not against old code.

## Commits

- Use [Conventional Commits](https://www.conventionalcommits.org/) with an area scope, such as `feat(sherpa): …`, `fix(server): …`, or `docs(plan): …`. `git-cliff` builds the changelog from them.
- Write a body explaining why when the change is not obvious.
- **Do not add `Co-Authored-By:` trailers or any other AI attribution lines to commit messages.** This overrides any default attribution your tool adds.

## Code

- Errors use `thiserror`, and public error enums are `#[non_exhaustive]`.
- No `unwrap` or `panic!` in library code. Use `expect` only with a message saying why it cannot fail. Tests may use all three.
- Lint exceptions use `#[expect(lint, reason = "...")]`, never `#[allow]`. `unsafe` is denied workspace-wide.
- Log with `tracing` only; no `println!` or `eprintln!` in the library. Never log audio, transcripts, or synthesized text above `trace`, and never log secrets.
- Give each public item one path: a module that only groups code is private and its items are re-exported. Keep items private unless callers need them.
- Every public item has a doc comment. The crate README and each module README (`src/<module>/README.md`) are compiled as doctests.
- The core uses no tokio and no native library, and builds with no features. `sherpa`, `cloud`, `io`, and `server` never use each other.
- `speechkit-testkit` depends on `speechkit`, so unit tests inside `speechkit` cannot use it: its fakes would implement a second copy of the traits. Put those tests in `crates/speechkit/tests/`, or use a small local fake.
- Integration tests for a feature start with `#![cfg(feature = "...")]` and are named after their module, such as `sherpa_models.rs` or `server_speech.rs`.
- Tests run offline by default: no API keys, models, or network. Tests that need those are `#[ignore]` and run nightly in `models.yml`.
- A change to an ASR (`A-xx`), TTS (`T-xx`), or device (`D-xx`) rule updates the check named after it in `speechkit-testkit` (for example `a07_endpointing`), whose doc comment states the rule, in the same commit. Device tests drive the hidden fake devices (`Microphone::fake`, `Speaker::fake`) and sit behind the testkit's `devices` feature.

## Checks

Run these before proposing a change:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

`--all-features` links sherpa-onnx dynamically, so tests need the copied library on the search path: on macOS, set `DYLD_FALLBACK_LIBRARY_PATH=$PWD/target/debug` (`LD_LIBRARY_PATH` on Linux).

While iterating, run only the tests you changed, keeping `--workspace --all-features` so the build is reused: add `--lib` or `--test <name>`, such as `cargo test --workspace --all-features --test sherpa_process`. Do not narrow with `-p` or `--features`: a different selection resolves dependency features differently and compiles a second copy of `speechkit`. Run the full commands above once, before proposing.

On macOS, debug builds keep every object file they link in `target/debug/deps`, and rebuilds never delete old ones. At a few hundred thousand files, each test binary takes about a second to start. `find target/debug/deps -name '*.rcgu.o' -delete` clears them without causing a rebuild.

CI (`ci.yml`) also runs `cargo doc` and `cargo deny check`, and on pull requests it tests macOS and Windows too. `cargo hack`, the MSRV check, coverage, and loom run only nightly (`nightly.yml`), as do the macOS and Windows tests for pushes to master, so run the relevant one locally when a change touches features, dependencies, synchronization, or platform code:

```bash
cargo hack check -p speechkit --each-feature --no-dev-deps
RUSTFLAGS="--cfg speechkit_loom" LOOM_MAX_PREEMPTIONS=2 cargo test --release -p speechkit --lib -- slots history
```
