use super::*;

const TRANSDUCER: &[&str] = &[
    "encoder-epoch-99-avg-1.int8.onnx",
    "encoder-epoch-99-avg-1.onnx",
    "decoder-epoch-99-avg-1.int8.onnx",
    "decoder-epoch-99-avg-1.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
    "joiner-epoch-99-avg-1.onnx",
    "tokens.txt",
    "bpe.model",
    "bpe.vocab",
    "test_wavs/0.wav",
];

const SENSE_VOICE: &[&str] = &[
    "model.int8.onnx",
    "model.onnx",
    "tokens.txt",
    "test_wavs/en.wav",
];

const QWEN3: &[&str] = &[
    "conv_frontend.onnx",
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "tokenizer/merges.txt",
    "tokenizer/vocab.json",
];

const FUNASR: &[&str] = &[
    "encoder_adaptor.int8.onnx",
    "llm.int8.onnx",
    "embedding.int8.onnx",
    "Qwen3-0.6B/vocab.json",
    "Qwen3-0.6B/merges.txt",
    "Qwen3-0.6B/tokenizer.json",
    "test_wavs/zh.wav",
];

const FIRE_RED_AED: &[&str] = &["encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"];

fn wrapped(files: &[&str]) -> Vec<String> {
    files.iter().map(|f| format!("archive-2024/{f}")).collect()
}

fn error(result: Result<impl std::fmt::Debug, SpeechError>) -> String {
    match result {
        Err(SpeechError::InvalidModel(message)) => message,
        other => panic!("expected InvalidModel, got {other:?}"),
    }
}

/// One row per known archive layout.
#[test]
fn detection_table() {
    let rows: &[(&str, &[&str], Result<AsrModelLayout, &str>)] = &[
        (
            "streaming zipformer bilingual",
            TRANSDUCER,
            Ok(AsrModelLayout::Transducer),
        ),
        (
            "plain transducer names",
            &["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"],
            Ok(AsrModelLayout::Transducer),
        ),
        ("sense voice", SENSE_VOICE, Ok(AsrModelLayout::Flat)),
        (
            "paraformer",
            &["model.int8.onnx", "tokens.txt"],
            Ok(AsrModelLayout::Flat),
        ),
        (
            "firered ctc",
            &["model.onnx", "tokens.txt"],
            Ok(AsrModelLayout::Flat),
        ),
        ("qwen3", QWEN3, Ok(AsrModelLayout::Qwen3Asr)),
        ("funasr nano", FUNASR, Ok(AsrModelLayout::FunAsrNano)),
        ("firered aed", FIRE_RED_AED, Ok(AsrModelLayout::FireRedAed)),
        (
            "transducer without tokens",
            &["encoder.onnx", "decoder.onnx", "joiner.onnx"],
            Err("missing tokens.txt"),
        ),
        (
            "transducer without decoder",
            &["encoder.onnx", "joiner.onnx", "tokens.txt"],
            Err("missing decoder*.onnx"),
        ),
        (
            "qwen3 without tokenizer",
            &["conv_frontend.onnx", "encoder.onnx", "decoder.onnx"],
            Err("missing tokenizer/merges.txt"),
        ),
        (
            "qwen3 with half a tokenizer",
            &[
                "conv_frontend.onnx",
                "encoder.onnx",
                "decoder.onnx",
                "tokenizer/merges.txt",
            ],
            Err("missing tokenizer/vocab.json"),
        ),
        (
            "funasr without tokenizer",
            &["encoder_adaptor.onnx", "llm.onnx", "embedding.onnx"],
            Err("missing a tokenizer directory"),
        ),
        (
            "funasr without llm",
            &[
                "encoder_adaptor.onnx",
                "embedding.onnx",
                "t/vocab.json",
                "t/merges.txt",
                "t/tokenizer.json",
            ],
            Err("missing llm*.onnx"),
        ),
        (
            "firered aed without tokens",
            &["encoder.onnx", "decoder.onnx"],
            Err("missing tokens.txt"),
        ),
        ("model without tokens", &["model.onnx"], Err("unrecognized")),
        ("empty directory", &[], Err("unrecognized")),
        (
            "unrelated files",
            &["README.md", "notes.txt"],
            Err("unrecognized"),
        ),
    ];
    for (name, files, expected) in rows {
        let result = detect_asr(files);
        match expected {
            Ok(layout) => assert_eq!(result.ok(), Some(*layout), "{name}"),
            Err(fragment) => {
                let message = error(result);
                assert!(message.contains(fragment), "{name}: {message}");
            }
        }
        let inside = wrapped(files);
        let again = detect_asr(&inside);
        assert_eq!(again.is_ok(), expected.is_ok(), "{name}, wrapped");
    }
}

#[test]
fn transducer_prefers_int8_encoder_and_joiner_but_float_decoder() {
    let files = select_asr(TRANSDUCER, AsrModelLayout::Transducer).unwrap();
    assert_eq!(
        files,
        AsrFiles::Transducer {
            encoder: "encoder-epoch-99-avg-1.int8.onnx".into(),
            decoder: "decoder-epoch-99-avg-1.onnx".into(),
            joiner: "joiner-epoch-99-avg-1.int8.onnx".into(),
            tokens: "tokens.txt".into(),
            bpe_vocab: Some("bpe.vocab".into()),
        }
    );
    let only_int8 = [
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ];
    let AsrFiles::Transducer {
        decoder, bpe_vocab, ..
    } = select_asr(&only_int8, AsrModelLayout::Transducer).unwrap()
    else {
        panic!("transducer files");
    };
    assert_eq!(decoder, "decoder.int8.onnx");
    assert_eq!(bpe_vocab, None);
}

#[test]
fn wrapper_directory_is_kept_in_paths() {
    let inside = wrapped(SENSE_VOICE);
    let files = select_asr(&inside, AsrModelLayout::SenseVoice).unwrap();
    assert_eq!(
        files,
        AsrFiles::Flat {
            model: "archive-2024/model.int8.onnx".into(),
            tokens: "archive-2024/tokens.txt".into(),
        }
    );
    let inside = wrapped(FUNASR);
    let AsrFiles::FunAsrNano { tokenizer, llm, .. } =
        select_asr(&inside, AsrModelLayout::FunAsrNano).unwrap()
    else {
        panic!("funasr files");
    };
    assert_eq!(tokenizer, "archive-2024/Qwen3-0.6B");
    assert_eq!(llm, "archive-2024/llm.int8.onnx");
    let inside = wrapped(QWEN3);
    let AsrFiles::Qwen3Asr {
        tokenizer,
        conv_frontend,
        ..
    } = select_asr(&inside, AsrModelLayout::Qwen3Asr).unwrap()
    else {
        panic!("qwen3 files");
    };
    assert_eq!(tokenizer, "archive-2024/tokenizer");
    assert_eq!(conv_frontend, "archive-2024/conv_frontend.onnx");
}

/// Regression: a full transducer directory configured as
/// FireRed-AED is refused before any native call.
#[test]
fn transducer_directory_is_not_fire_red_aed() {
    for joiner in [
        "joiner.int8.onnx",
        "joiner-epoch-99-avg-1.onnx",
        "joiner.fp16.onnx",
    ] {
        let files = ["encoder.int8.onnx", "decoder.onnx", joiner, "tokens.txt"];
        let message = error(select_asr(&files, AsrModelLayout::FireRedAed));
        assert!(
            message.contains("contains joiner*.onnx; this is a transducer layout"),
            "{message}"
        );
        assert_eq!(detect_asr(&files).unwrap(), AsrModelLayout::Transducer);
    }
    let files = select_asr(FIRE_RED_AED, AsrModelLayout::FireRedAed).unwrap();
    assert_eq!(
        files,
        AsrFiles::FireRedAed {
            encoder: "encoder.int8.onnx".into(),
            decoder: "decoder.int8.onnx".into(),
            tokens: "tokens.txt".into(),
        }
    );
}

#[test]
fn explicit_family_must_match_the_files() {
    assert!(error(select_asr(SENSE_VOICE, AsrModelLayout::Transducer)).contains("encoder"));
    assert!(error(select_asr(TRANSDUCER, AsrModelLayout::Flat)).contains("model*.onnx"));
    assert!(error(select_asr(FIRE_RED_AED, AsrModelLayout::Qwen3Asr)).contains("conv_frontend"));
    assert!(error(select_asr(QWEN3, AsrModelLayout::FunAsrNano)).contains("encoder_adaptor"));
}

#[test]
fn ambiguous_candidates_are_named() {
    let files = [
        "encoder-a.int8.onnx",
        "encoder-b.int8.onnx",
        "decoder.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    ];
    let message = error(select_asr(&files, AsrModelLayout::Transducer));
    assert!(
        message.contains("encoder-a.int8.onnx, encoder-b.int8.onnx"),
        "{message}"
    );
    assert!(message.contains("keep exactly one"), "{message}");
    let two_tokenizers: Vec<&str> = FUNASR
        .iter()
        .copied()
        .chain([
            "other/vocab.json",
            "other/merges.txt",
            "other/tokenizer.json",
        ])
        .collect();
    let message = error(select_asr(&two_tokenizers, AsrModelLayout::FunAsrNano));
    assert!(
        message.contains("several tokenizer directories"),
        "{message}"
    );
}

#[test]
fn sense_voice_markers() {
    let many = format!("{}<|zh|> 24884\n", "word 1\n".repeat(24_884));
    assert!(is_sense_voice_tokens(&many));
    assert!(is_sense_voice_tokens("<|en|> 1\n<|zh|> 2\n"));
    assert!(!is_sense_voice_tokens("IQ== 1\nPGJsaz4= 60514\n"));
    assert!(!is_sense_voice_tokens("x <|zh|>\n"));
}

#[test]
fn punctuation_layouts() {
    let ct = ["model.onnx", "model.int8.onnx", "tokens.json"];
    assert_eq!(
        select_punct(&ct).unwrap().layout,
        PunctuationFamily::CtTransformer
    );
    assert_eq!(select_punct(&ct).unwrap().model, "model.int8.onnx");
    let en = ["model.int8.onnx", "bpe.vocab"];
    let files = select_punct(&en).unwrap();
    assert_eq!(files.layout, PunctuationFamily::CnnBiLstm);
    assert_eq!(files.vocab.as_deref(), Some("bpe.vocab"));
    let inside = wrapped(&["model.int8.onnx"]);
    assert_eq!(
        select_punct(&inside).unwrap().model,
        "archive-2024/model.int8.onnx"
    );
    assert!(error(select_punct(&["bpe.vocab"])).contains("model*.onnx"));
}

#[test]
fn layout_names() {
    for layout in [
        AsrModelLayout::Transducer,
        AsrModelLayout::SenseVoice,
        AsrModelLayout::Flat,
        AsrModelLayout::Qwen3Asr,
        AsrModelLayout::FunAsrNano,
        AsrModelLayout::FireRedAed,
    ] {
        assert!(!layout.to_string().is_empty());
    }
}

#[test]
fn list_dir_walks_and_rejects_empty_model_files() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    std::fs::create_dir_all(root.join("tokenizer/deeper/too-deep")).unwrap();
    for name in [
        "encoder.onnx",
        "tokens.txt",
        ".hidden",
        "tokenizer/vocab.json",
    ] {
        std::fs::write(root.join(name), "x").unwrap();
    }
    std::fs::write(root.join("tokenizer/deeper/too-deep/far.txt"), "x").unwrap();
    std::fs::write(root.join("README.md"), "").unwrap();
    let files = list_dir(dir.path()).unwrap();
    assert_eq!(
        files,
        [
            "archive/README.md",
            "archive/encoder.onnx",
            "archive/tokenizer/vocab.json",
            "archive/tokens.txt",
        ]
    );
    std::fs::write(root.join("decoder.onnx"), "").unwrap();
    let message = error(list_dir(dir.path()));
    assert!(
        message.contains("archive/decoder.onnx is empty"),
        "{message}"
    );
    let message = error(list_dir(&dir.path().join("missing")));
    assert!(message.contains("cannot read model directory"), "{message}");
}

#[cfg(unix)]
#[test]
fn list_dir_skips_entries_that_are_neither_files_nor_directories() {
    let dir = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(dir.path().join("nowhere"), dir.path().join("dangling")).unwrap();
    std::fs::write(dir.path().join("tokens.txt"), "x").unwrap();
    assert_eq!(list_dir(dir.path()).unwrap(), ["tokens.txt"]);
}
