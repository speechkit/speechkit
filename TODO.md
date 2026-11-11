# TODO

Findings from the codebase review of 2026-11-10 (at commit 976c662), and a proposal for what to do about them and around them.

Each item says how sure it is:

- **Confirmed**: reproduced by running something.
- **Read**: found by reading the code; not yet reproduced. Write the failing test first.

## State of the checks

- `cargo fmt --all -- --check`: passes.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: passes.
- `cargo test --workspace --all-features --no-fail-fast`: 494 passed, 1 failed, 43 ignored at review time; the failure was item 1. After Phase 1: 498 passed, 0 failed, 43 ignored. After Phase 2: 510 passed, 0 failed, 43 ignored.
- After Phase 2, `cargo doc` with `-D warnings` and `cargo hack check -p speechkit --each-feature --no-dev-deps` also pass.
- Not run yet: `cargo deny check`, the loom tests, the `#[ignore]` model tests. Item 7 changes session synchronization, but not the `slots` or `history` code the loom tests model.

## Review findings

### 1. `d02_behind_fails` is flaky (Confirmed; fixed 2026-11-11)

- [x] Make the overflow half of the test deterministic.
- [x] Fix the stop race behind its last assertion.

It was two races, not one:

1. **Test side.** `FakeMicrophone::push_burst` pushed 10 s into the 2 s ring one sample at a time while the capture thread drained it every 20 ms. On a slow run nothing was lost and the check timed out. It now takes the ring's free space once (`Device::input_within`), so whatever is past it is always lost. Failed about 4 runs in 7 before.
2. **Product side.** With that fixed, about 1 run in 60 still failed: a listening stopped just after `push` heard 4,625 of 4,800 samples. `Listening::stop` resolved to "the capture position when the capture thread next runs". If the stop landed between that thread draining the ring and appending, samples delivered *before* the stop were left out. That broke the documented promise, and it applies to real microphones too: a stop could drop up to a tick of audio. A stop now records `Device::delivered()`, the number of samples the callback has queued, which is exactly a timeline position. `Stop::Requested` is gone; a stop may be ahead of the timeline's end, and readers wait for the capture thread to catch up to it.

Verified: the device suites passed 400 sequential runs (100 per binary) and 120 runs of `io_contract` at 8 in parallel, plus 80 loaded runs each of the other device binaries and the io unit tests. New unit tests cover the capped callback, a stop ahead of the moved audio, a stop past an ended timeline, and the delivered count.

### 2. `audio::read` rejects the WAV the server produces (Confirmed; fixed in 72c3ebe)

- [x] Read WAV files whose size fields are `0xFFFFFFFF`.

Done as planned for `0xFFFFFFFF`. A data length past the end of the source is not special-cased, and there is no test for it yet. The `server_speech.rs` WAV test now decodes the server's output with `audio::read`. Still to do: add the header to the `read_wav_bytes` fuzz corpus.

`POST /v1/audio/speech` answers `wav` by default with a streaming header whose RIFF and data sizes are `0xFFFFFFFF` (`crates/speechkit/src/server/speech.rs:119`). hound refuses it ("data chunk length is not a multiple of sample size"), and `from_source` (`crates/speechkit/src/audio/mod.rs:142`) returns that as `Invalid`, so symphonia is never tried. The same header comes from `ffmpeg -f wav -`.

So `speechkit transcribe`, `audio::read`, `audio::decode`, and the server's own transcription route all reject audio that speechkit's server wrote.

Fix: in `wav::read`, treat a data length of `0xFFFFFFFF` (and a length past the end of the source) as "until the end", still bounded by `DecodeLimits::max_duration` while reading. Add a round-trip test that feeds `/v1/audio/speech` output to `audio::decode`, and add the header to the `read_wav_bytes` fuzz corpus.

### 3. A worker that hangs before its handshake is never killed (Confirmed; fixed in 5822b91)

- [x] Bound worker startup with a timeout.
- [ ] `IsolatedAsr::spawn` builds its own `WorkerCommand`, so its callers get the 120 s default and cannot change it. Add a way, or document `spawn_command` as the way.
- [ ] The blocking write in `Worker::call`, below, is not fixed; it belongs to Phase 4's bounded-waits audit.

Reproduced with a fake worker that stalls before its handshake: `spawn_command` blocked, and so did a session that had to start a new worker. `WorkerCommand::with_start_timeout` (120 s by default) now bounds the wait, and a worker that passes it is killed with a retryable error. SECURITY.md says so.

`Worker::start` (`crates/speechkit/src/sherpa/process/client.rs:112`) reads the handshake with a blocking read and no timeout. The worker loads its model before it prints the handshake, so a hang in the native loader, which is what `IsolatedAsr` exists to contain, blocks forever.

- In `open`, the session fails on its deadline, but its thread stays in the read, holds its slot, and the child stays alive.
- In `IsolatedAsr::spawn_command`, the caller itself blocks forever.

The struct doc says a new worker "loads the model within `start`'s deadline"; nothing enforces that.

Also, `Worker::call` writes the request with a blocking write. A frame larger than the pipe buffer to a worker that has stopped reading blocks before the reply timeout starts.

Fix: start the reader thread before the handshake and receive the hello through the reply channel with a timeout (a `with_start_timeout`, defaulting to something generous for model loads); kill the child when it passes. Test with a worker command that sleeps before its handshake (`sherpa_process.rs`).

### 4. A streaming transcription client that stops reading holds an admission permit (Confirmed; fixed in 774a623)

- [x] Give the SSE path a send deadline.

Reproduced with a client that holds an unread `stream=true` response. A deadline alone was not enough: the permit lived in the response stream's state, and a body nobody polls is never dropped. The permit now belongs to the blocking thread that forwards the events. SECURITY.md is updated.

`forward` (`crates/speechkit/src/server/transcriptions.rs:321`) calls `blocking_send` with no deadline. When a client stops reading a `stream=true` response larger than the socket buffers, the channel fills, the blocking thread waits forever, and the admission permit, which lives in the response stream's state, is never released. As many such connections as the engine has session slots (8 by default) make every transcription request answer 503 until those connections die.

`/v1/audio/speech` already handles this: `pump` sends with `timeout_at(deadline, ...)`. SECURITY.md claims the defence only for synthesis.

Fix: pass the request deadline to `forward` and send with it, as `pump` does; when it passes, cancel the session and return, so the stream ends and the guard drops. Then update the SECURITY.md row. Test in `server.rs` with a client that never reads.

### 5. `serve --bind localhost:8080` cannot work (Confirmed; fixed in 89d3c4b)

- [x] Resolve host names in `Server::start`, or reject them before the model loads.

Both: `start` resolves and binds the first address that works, and the CLI resolves `--bind` before it builds the engine, so a bad name is a usage error. Note for tests: this machine's resolver answers every name, even `*.invalid`, so no test relies on a name failing to resolve.

Commit 976c662 stopped the missing-authentication warning for names that resolve to loopback, but `Server::start` (`crates/speechkit/src/server/mod.rs:72`) parses the address as a literal `SocketAddr`. The user waits for the model to load, then gets "invalid bind address".

Fix (preferred): resolve with `tokio::net::lookup_host` in `start` and bind the first address that works; document that `with_bind` takes `host:port`. Otherwise, validate `--bind` in the CLI before building the engine. Either way, add a CLI test that serves on `localhost:0`.

### 6. A CLI timeout prints "cancelled" (Confirmed; fixed in 8312714)

- [x] Report `DeadlineExceeded` when `--timeout` passes.

`show_until_done` (`crates/speechkit-cli/src/progress.rs:70`) calls `session.cancel()` when its deadline passes, so `transcribe` and `stream` end with `error: cancelled`.

Fix: call `session.finish(deadline)` there instead; with the deadline passed it settles the session as `DeadlineExceeded` at once. Add a CLI test with a fake backend that never finishes and `--timeout 1`.

### 7. Late events are recorded after a session has failed (Confirmed; fixed in 3da75ee)

- [x] Keep the session "stopping" until its result is published.

Reproduced through both paths, a failed call and a failure reported between calls: the late segment reached readers and the failure's confirmed transcript. The fake gained `Trigger::OnCancel`. The new check is `a03_late_events_after_a_failure_discarded`; the rule itself is unchanged.

`Running::run` (`crates/speechkit/src/asr/session/worker.rs:132`) takes `core.failure` before it cancels and drops the stream. From then until `settle_failure`, `Shared::stopping` is false again: events the stream sends from `cancel` or from its own threads are recorded and reach readers, and a push is accepted and then silently dropped. The comment beside it says late events are discarded (A-03). The same window exists when `drive` returns an error, where `failure` was never set.

Fix: leave the failure in place (clone it, or store the error from `drive` into it) and clear it only inside the final lock where the result is settled. Add an `asr` fault check to the testkit for a stream that sends a segment from `cancel`.

### Smaller notes

- [x] (69ded22) `require_auth` (`crates/speechkit/src/server/error.rs:75`) matches `Bearer ` case-sensitively; the scheme is case-insensitive in RFC 9110. Accept `bearer` too.
- [x] (9072ac8) `Server::with_auth(Secret::new(""))` accepts the header `Authorization: Bearer `. The CLI cannot produce it (`Secret::from_env` refuses an empty value), but the library can. Refuse an empty token in `start` and `router`. `router` cannot return an error, so with an empty token it answers 401 to every `/v1/` request.
- [x] (931edff) `RunningServer` stops the server when it is dropped (the shutdown sender drops). That is reasonable, but undocumented: `let _ = server.start(..).await?` stops at once. Say so in its doc comment, and mark it `#[must_use]`.
- [ ] `IsolatedAsr`'s idle pool only grows: after a burst it keeps one loaded model per peak concurrent session. Consider a cap or an idle timeout. Left open: it needs a design decision, not a fix.

### Phase 3 findings

Found on 2026-11-11 by reading the areas Phase 3 lists. Items 8 to 15 are not fixed yet; each needs its test first, as in Phase 2.

### 8. `Playback::played` counts audio before it is heard (Confirmed; D-06)

- [ ] Count only the samples whose playback time has passed, whatever the device latency.

`Shared::played_index` (`crates/speechkit/src/io/speaker/mod.rs:189`) knows only the latest callback buffer: its first index, when that buffer starts playing, and its length. When the callback for buffer N+1 runs, buffer N is usually still waiting to be played, since device latency is longer than one buffer on most devices (tens of ms; Bluetooth, 150 ms or more). But `played_index` already returns the start of buffer N+1. So `played()` and `text_played()` run ahead of what the listener heard by about the latency, which is what D-06 rules out and what barge-in relies on.

Reproduced with a temporary unit test: two 10 ms buffers rendered with 100 ms latency, then `played_index()` returned 160 frames when nothing had reached its playback time. The device contract checks cannot see it, because `FakeSpeaker::play` renders with zero latency.

Fix: when the latest buffer's time is still ahead, count back from its start by `frames_in(at - now)`, or keep a short history of buffers, and never below the last index already reported. Give `FakeSpeaker` a latency setting, and add the check to D-06.

### 9. Dropping a cloud stream can block on a peer that stopped reading (Read)

- [ ] Bound the whole close in `Pushed::drop`.

`Pushed::drop` (`crates/speechkit/src/cloud/ws.rs:552`) sends the close frame within `CLOSE_WAIT` (1 s). It then calls `sink.close().await` with no timeout, and that flushes whatever is still buffered. If the server has stopped reading and the socket buffers are full, for example after an `append` that timed out, the drop waits until the TCP connection dies. The session's slot is held until the drop returns, which is the item 3 and 4 pattern again.

Fix: put the whole close sequence under `timeout(CLOSE_WAIT, …)`. Test with a WebSocket server that accepts and then never reads.

### 10. Keyword spotting after a pause may report wrong times, or miss a repeat (Read; needs a model)

- [ ] Check a keyword, then 3 s of silence, then the keyword again, against the kws-en model.

`Detector::decode` (`crates/speechkit/src/sherpa/kws/mod.rs:473`) never resets the stream. It relies on two things: sherpa-onnx's decoder restarting after a match, and `GetKeywordResult(true)` dropping a result that overlaps the last one. Upstream source (sherpa-onnx master, read 2026-11-11) confirms both. But `KeywordSpotterTransducerImpl::DecodeStreams` also resets the stream by itself after 1.5 s of trailing blanks. That reset re-initializes the decoder result, which probably restarts token timestamps. It does not clear `prev_keyword_result_`, which the duplicate check compares against. So after a pause:

- a new detection may be dated from the reset, not from the start of the stream;
- or a genuine second keyword may be dropped as a "duplicate", because its first timestamp is below the previous keyword's last one.

This is inferred from the source, not run: no model is on this machine. `a_repeated_keyword_is_spotted_each_time` concatenates one clip three times, and may not hold a 1.5 s pause. Upstream's own examples call `reset` after every detection.

Fix, if it reproduces: reset after each detection, add the samples fed before the reset to the token times, and drop the comment's claim that token times always count from the start of the stream.

### 11. A sound queued just after `Speaker::stop` can be stopped by it (Read; minor)

- [ ] Make `Speaker::stop` stop only what was queued before it.

`Speaker::stop` drains the queue and sets `stop_all`, and the player acts on the flag in its next `service`. The player calls `service`, then `next_sound`. If `stop` and a new `play` both land between those two calls, the player takes the new sound and stops it on its next pass. Its `finish` then returns `Ok` with nothing played. The window is microseconds wide.

Fix: number the sounds as they are queued, and have `stop` record the last number it covers.

### 12. A streaming transducer can date a segment past the audio it was given (Read; minor)

- [ ] Clamp token times to the samples fed.

`Stream::decode` (`crates/speechkit/src/sherpa/asr/streaming.rs:283`) turns token timestamps into segment times as they are. `finish` feeds `TAIL_PADDING` of silence, so a token decoded in the padding has a time past the end of the input. The keyword spotter clamps its times to the audio heard (`seconds(value, heard)`); the streaming recognizer does not.

Fix: clamp to `at(self.fed)`, as the spotter does, and assert `end <= duration` in the model tests, as `sherpa_kws.rs` does.

### 13. The release workflow trusts the tag (Read)

- [ ] Decide which of these to adopt.

`release.yml` publishes to crates.io on any `v*` tag that matches the manifest and the changelog. Nothing checks that CI passed on the tagged commit; `cargo xtask release-check` covers part of it, but only if someone runs it first. The actions are pinned by tag (`@v4`, `@v1`, `@stable`), not by commit, in jobs that hold `id-token: write` and `contents: write`. The release attaches binaries without checksums or provenance.

Options: a job that checks the `ci.yml` run for the tagged commit; actions pinned to commit SHAs (with Dependabot to bump them); a `SHA256SUMS` file, or `actions/attest-build-provenance`.

### 14. `xtask` unpacks a download before it knows its size is right (Read; minor)

- [ ] Stop reading past the manifest's size.

`fetch_archive` (`xtask/src/models.rs:317`) checks the archive's size and hash only after the last byte, as documented. Members cannot escape the staging directory (`unpack_in`). But a bad mirror or a compression bomb can write any amount to disk first. Fail the `Hashing` reader as soon as it passes `archive.size`; the hash check stays at the end.

### 15. More checks that assume a fast machine (Read)

- [ ] Watch these in the nightly repeats, with the two Phase 1 named.

- `t02_finish_waits_for_the_reader` (`speechkit-testkit/src/contract/tts/faults.rs:73`) expects audio to have been synthesized within a 200 ms `finish`.
- The device checks use `AT_ONCE` (100 ms) as an upper bound for `stop` and for a cancelled finish (`contract/devices.rs:403`, `:436`).

The other sleeps in the testkit are followed by upper bounds or "nothing happened" checks.

### Read, nothing found

The OpenAI SSE decoder and HTTP limits; the Realtime and DashScope state machines and the shared WebSocket envelope (apart from item 9); the keyword tokenizer; the offline recognizer's checks; layout selection; silero VAD; sherpa TTS; bias and punctuation; `tts/chunk`; `asr/engine`; `vad/energy`; `speech/resample`; `release-check`.

## Proposal

In order. The first two phases are small and should each be a handful of `fix(...)` commits; the later ones are larger and need a decision before work starts.

### Phase 1: make the checks trustworthy (done 2026-11-11)

- [x] Fix item 1.
- [x] `--no-fail-fast` everywhere tests run. `ci.yml` already had it; the review was wrong there. Nightly's macOS/Windows `test` job and its `test-minimal` step did not, and now do.
- [x] Nightly repeats the four device test binaries 50 times on macOS and Windows, after the full test run and reusing its build. Not yet seen running in Actions: check the first nightly after this lands.
- [x] Audit the other device tests for the same assumption. Most fixed sleeps are followed by upper bounds or "nothing happened" checks, which slowness cannot break. Two expect something to have happened in time, and passed 80 loaded runs each, so they are left as they are; watch them in the nightly repeats:
  - `io_scenarios.rs:511` expects the player to have skipped a reply within 100 ms.
  - `speaker/tests.rs:271` expects the producer to be waiting after 100 ms; it already says "rerun on a less loaded machine" when not.

### Phase 2: fix the findings (done 2026-11-11)

- [x] Items 2 to 7 and the smaller notes, one commit each, test first. Items 3 to 7 were found by reading, so the test is also the confirmation. Every item reproduced, so none was dropped. The pool cap is left open on purpose, as are the follow-ups listed under items 2 and 3.
- [x] Update SECURITY.md where a fix changes what it claims: items 3 and 4, and the empty auth token.

### Phase 3: finish the review (done 2026-11-11; findings 8 to 15)

The review did not read these. They are the larger part of the code by lines, and two of them parse untrusted input:

- [x] `io/speaker` (about 1,600 lines with tests): playback, marks, the "played" accounting behind D-06.
- [x] `cloud/openai/realtime` and `cloud/dashscope` protocol state: everything a server can send, in any order.
- [x] `cloud/openai/http`: the SSE parser and the byte limits SECURITY.md describes.
- [x] `sherpa`: `kws` and `kws/tokenize`, `asr/recognizer`, `asr/streaming`, `layout`, `tts`, `vad`, `bias`, `punct`.
- [x] `tts/chunk`, `asr/engine`, `vad/energy`, `speech/resample`.
- [x] `xtask` (model fetching and hashing, release) and the release workflow.
- [x] The `speechkit-testkit` fakes, for more timing assumptions like item 1.

### Phase 4: close the gaps the findings point at

Each finding is one case of a wider gap. These are worth doing once, rather than waiting for the next case.

- [ ] **Every blocking wait on something outside the process has a bound.** Items 3, 4 and 9 are each an unbounded wait on a peer (a child process, a client socket, a WebSocket server). Grep for `blocking_send`, `read_exact`, `write_all`, `wait_forever`, and `recv()` without a timeout, and write down for each why it cannot block forever, or bound it.
- [ ] **Round-trip tests across speechkit's own boundaries.** Item 2 is the server writing something the library cannot read. Add tests that pipe each producer into each consumer: `/v1/audio/speech` → `audio::decode` → `/v1/audio/transcriptions`; `encode_wav` → `audio::read`; the OpenAI speech client against the speechkit server. `round_trip.rs` is the place.
- [ ] **CLI tests for failure messages.** Items 5 and 6 are both the CLI saying the wrong thing on a failure path. Add a table test of failures (timeout, bad bind, missing model, bad audio) with the exit code and the message each must give.
- [ ] **A server fuzz or soak target for slow clients.** `server_request` fuzzes request bodies; nothing exercises slow or stalled readers and writers. A test harness with a throttled socket would have caught item 4.
- [ ] **Write-side connection limits.** hyper has no write timeout, so a stalled reader keeps its connection open even after item 4. SECURITY.md already leaves connection limits to a proxy; decide whether an idle or total response timeout belongs in the server, and document the decision either way.

### Phase 5: worth considering, not urgent

- [ ] `cargo nextest` in CI: per-test timeouts and retries reported as flaky rather than hidden, and faster start on macOS.
- [ ] A `speechkit doctor` or `serve --check` that loads the model, binds, and exits, so a deployment can tell a bad flag from a slow load before taking traffic.
- [ ] Health reporting for `IsolatedAsr`: worker starts, crashes, and kills as counters, exposed on `/health`, so a crash loop is visible.
