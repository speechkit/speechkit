# Accuracy evaluation

How to measure how accurate recognition is. This is separate from the
golden tests in `crates/speechkit/tests/sherpa_models.rs`, which pin each
backend's output on the clips bundled with the models and only catch
*change*; an accuracy evaluation scores transcripts against human
references on a corpus and answers how wrong a backend is, and where.

## Metrics

`speechkit_testkit::metrics` computes the rates:

- `cer` — character error rate, the primary metric for Chinese and for
  mixed Chinese-English text.
- `wer` — word error rate, for English-only text. On Chinese it scores
  whole utterances as single words and is meaningless.
- `char_errors` and `word_errors` — the substitution, deletion, and
  insertion breakdown behind each rate.

The normalization policy and how to read the breakdown are documented on
the module. In short: case is folded and everything that is not a letter
or digit is dropped, digits compare as written ("123" and "一百二十三"
are wholly different), deletions clustered at utterance ends point at
endpointing or VAD cutting speech early, insertions at utterance
boundaries point at VAD merging utterances, and substitutions spread
across the utterance are acoustic or model errors.

## Datasets

`fixtures/evals.json` lists evaluation corpora; `cargo xtask
fetch-evals` downloads, verifies, and unpacks them into a cache of
their own:

```sh
cargo xtask fetch-evals                  # everything
cargo xtask fetch-evals --only aishell-1 # one dataset
```

They land in `$SPEECHKIT_EVALS` (`~/.cache/speechkit/evals` by
default) and are found through `SPEECHKIT_EVAL_<ID>`; the models stay
in `$SPEECHKIT_MODELS`, so a corpus never sits among them. They are
kept in a separate file from `fixtures/manifest.json` so the nightly
Models workflow never downloads them; fetching is opt-in.

### aishell-1

[AISHELL-1](https://www.openslr.org/33): Mandarin, 400 speakers from
different accent regions, 16 kHz, quiet indoor recordings, Apache-2.0,
transcription accuracy above 95%.

The corpus ships its audio as one tarball per speaker, with the split
self-described inside each (`test/S0764/...`), so the entry uses
`nested` to unpack those in place and `include` to carry only the 20
test speakers listed in `fixtures/evals.json` — 7176 utterances, about
10 hours — never writing the training data. The full 15.6 GB archive is
still downloaded, because the sha256 check needs every byte. After
fetching:

```
$SPEECHKIT_EVAL_AISHELL_1/data_aishell/wav/test/S0764/BAC009S0764W0121.wav
$SPEECHKIT_EVAL_AISHELL_1/data_aishell/transcript/aishell_transcript_v0.8.txt
```

The transcript file maps every utterance ID to its reference text, one
per line.

### Mixed Chinese-English

No standard code-switching test set can be downloaded freely: SEAME and
ASCEND both require a license application. Until one is licensed:

- A licensed archive drops into `fixtures/evals.json` with no code
  changes; `include` and `nested` handle any tar layout.
- A synthetic switching condition can be generated from aishell-1 plus
  an English corpus: splice one Mandarin and one English utterance into
  one stream and score against the concatenated references. It measures
  stability at the switch — whether the transcript falls apart when the
  language changes mid-stream, which stresses VAD and streaming decoders
  — not natural code-switching accuracy. Report it as its own condition;
  never mix it into the headline CER.

## Models

The lineup in `crates/speechkit/tests/eval_asr.rs` has the small
published models of `fixtures/manifest.json` — fetched by
`fetch-fixtures` — and a large tier the manifest does not carry.
Download that tier by hand from the
[sherpa-onnx `asr-models` release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
and point `SPEECHKIT_MODEL_<ID>` at the unpacked directory; every entry
whose variable is set joins the evaluation:

| id | family | archive |
| --- | --- | --- |
| `funasr-nano` | FunAsrNano | `sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2` |
| `firered-aed` | FireRedAed | `sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26.tar.bz2` |
| `firered-ctc` | FireRedCtc | `sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2` |
| `qwen3-asr` | Qwen3Asr | `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2` |

Each archive sits at
`https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/<archive>`.
The `asr-models` tag is a rolling release: check it for newer archives
before fetching, and expect superseded assets to disappear. Record what
a published number was measured on.

## The harness

`crates/speechkit/tests/eval_asr.rs` runs the evaluation through
`AsrEngine` end to end — decode, resample, VAD segmentation,
endpointing — not straight to sherpa-onnx, because those stages change
the transcript and have knobs of their own:

```sh
cargo test -p speechkit --all-features --test eval_asr -- --ignored --nocapture
```

Every model in the test's lineup whose `SPEECHKIT_MODEL_<ID>` is set
joins the run; the rest print a skip note. It scores the first
`SPEECHKIT_EVAL_UTTERANCES` utterances of every test speaker (15 by
default; raise it toward 359 for the full split when a number matters)
against the human transcripts and prints the corpus CER with its
substitution, deletion, and insertion shares, the per-speaker CERs, and
the worst utterances. A decode or recognition failure scores as an
empty hypothesis rather than ending the run, and the test fails only on
a session error, never on a rate.

Streaming backends are fed each utterance as one buffered push, not at
real-time pace, so their endpointing differs from production: read
their CER as a fast-push control, not a deployed figure.

Not built yet:

- Feed streaming backends at real-time pace, so endpointing behaves as
  in production, with the fast-push numbers above as the control.
- Score per utterance, aggregate per speaker, and report a distribution
  (median and tail), not only a corpus mean.
- Pin provider and thread count in the report; CPU and CUDA differ.
- Hold out half the speakers if hotword lists are ever tuned against
  the corpus.

## Licensing

AISHELL-1 is Apache-2.0; credit the AISHELL foundation when publishing
numbers. The repository ships only URLs and hashes — the contents of
`fixtures/evals.json` — never corpus data.
