//! `cargo xtask fetch-fixtures`: download and verify the models listed in
//! `fixtures/manifest.json`. `cargo xtask fetch-evals` does the same for
//! the evaluation datasets in `fixtures/evals.json`, which are fetched
//! only on request because they are far larger than the models, into a
//! cache of their own so corpora never sit among the models.
//!
//! Each archive is streamed to disk while it is hashed, extracted into
//! `<dest>/.tmp-<random>`, checked file by file, and then renamed into
//! place, so a model directory is either complete and verified or absent.
//! A model's `extras` (files published separately, such as a Matcha
//! vocoder) are downloaded into its directory before it is checked.
//! An entry with `include` prefixes unpacks only the archive members
//! under one of them, so an evaluation set can carry a corpus's test
//! split without its training data; with `nested`, members that are
//! themselves tarballs (AISHELL-1 ships one per speaker) are unpacked
//! next to themselves and deleted.

use std::{
    fs::{self, File},
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{Result, workspace};

#[derive(Deserialize)]
struct Manifest {
    models: Vec<Model>,
}

#[derive(Deserialize)]
struct EvalManifest {
    evals: Vec<Model>,
}

#[derive(Deserialize)]
struct Model {
    id: String,
    name: String,
    archive: Archive,
    files: Vec<FileEntry>,
    #[serde(default)]
    extras: Vec<Extra>,
    /// Path prefixes; only archive members under one of them are
    /// unpacked. Empty unpacks everything.
    #[serde(default)]
    include: Vec<String>,
    /// Whether unpacked members that are themselves `.tar.gz` or
    /// `.tgz` archives are unpacked next to themselves and then
    /// deleted. Corpora such as AISHELL-1 ship one tar per speaker.
    /// Only members `include` lets through are considered, so it does
    /// nothing without `include`.
    #[serde(default)]
    nested: bool,
}

/// A file downloaded on its own and placed at `path` in the model
/// directory. List it in `files` too, so a present model is checked for it.
#[derive(Deserialize)]
struct Extra {
    path: String,
    #[serde(flatten)]
    download: Archive,
}

#[derive(Deserialize)]
struct Archive {
    url: String,
    sha256: String,
    size: u64,
}

#[derive(Deserialize)]
struct FileEntry {
    path: String,
    sha256: String,
}

/// Suffixes of archive members that `nested` unpacks in place.
const TARBALL_SUFFIXES: [&str; 2] = [".tar.gz", ".tgz"];

/// `$SPEECHKIT_MODELS`, or `~/.cache/speechkit/models`.
pub(crate) fn models_dir() -> PathBuf {
    cache_dir("SPEECHKIT_MODELS", "models")
}

/// `$SPEECHKIT_EVALS`, or `~/.cache/speechkit/evals`.
fn evals_dir() -> PathBuf {
    cache_dir("SPEECHKIT_EVALS", "evals")
}

/// `$SPEECHKIT_<NAME>`, or `~/.cache/speechkit/<name>`.
fn cache_dir(var: &str, name: &str) -> PathBuf {
    if let Some(dir) = std::env::var_os(var) {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(format!(".cache/speechkit/{name}"))
}

/// The environment variable a test reads for entry `id`: the kind
/// prefix, then the upper-cased ID with `-` read as `_`.
fn env_var(prefix: &str, id: &str) -> String {
    format!("{prefix}_{}", id.to_uppercase().replace('-', "_"))
}

fn read_json<T: serde::de::DeserializeOwned>(name: &str) -> Result<T> {
    let path = workspace().join(name);
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn load_manifest() -> Result<Manifest> {
    read_json("fixtures/manifest.json")
}

fn load_evals() -> Result<EvalManifest> {
    read_json("fixtures/evals.json")
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 16];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if read == 0 {
            return Ok(hex(&hasher.finalize()));
        }
        hasher.update(&buffer[..read]);
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// Whether `dir` holds every listed file with the right hash.
fn verify(model: &Model, dir: &Path) -> Result {
    if model.files.is_empty() {
        let hash = sha256_file(dir)?;
        return if hash == model.archive.sha256 {
            Ok(())
        } else {
            Err(format!("{}: sha256 {hash} does not match", dir.display()))
        };
    }
    for file in &model.files {
        let path = dir.join(&file.path);
        let hash = sha256_file(&path)?;
        if hash != file.sha256 {
            return Err(format!("{}: sha256 {hash} does not match", path.display()));
        }
    }
    Ok(())
}

/// Streams `url` into `out`, hashing it as it arrives.
fn download(archive: &Archive, out: &Path) -> Result {
    eprintln!(
        "downloading {} ({} MB)",
        archive.url,
        archive.size / 1_000_000
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .map_err(|e| e.to_string())?;
    let mut response = client
        .get(&archive.url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|e| format!("{}: {e}", archive.url))?;
    let mut file = File::create(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 16];
    let mut size = 0_u64;
    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|e| format!("{}: {e}", archive.url))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        file.write_all(&buffer[..read])
            .map_err(|e| format!("{}: {e}", out.display()))?;
        size += read as u64;
    }
    let hash = hex(&hasher.finalize());
    if size != archive.size || hash != archive.sha256 {
        return Err(format!(
            "{}: got {size} bytes with sha256 {hash}, expected {} bytes with {}",
            archive.url, archive.size, archive.sha256
        ));
    }
    Ok(())
}

fn random_suffix() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    format!("{}-{nanos}", std::process::id())
}

/// Extracts `model`'s downloaded archive into `staging` and returns
/// the unpacked root: the archive's top-level directory for a tar,
/// the file itself for anything else.
fn unpack(model: &Model, download_path: &Path, staging: &Path) -> Result<PathBuf> {
    let url = &model.archive.url;
    let file = || File::open(download_path).map_err(|e| e.to_string());
    let decoder: Box<dyn Read> = if url.ends_with(".tar.bz2") {
        Box::new(bzip2::read::BzDecoder::new(BufReader::new(file()?)))
    } else if TARBALL_SUFFIXES.iter().any(|s| url.ends_with(s)) {
        Box::new(flate2::read::GzDecoder::new(BufReader::new(file()?)))
    } else {
        return Ok(download_path.to_path_buf());
    };
    let mut archive = tar::Archive::new(decoder);
    if model.include.is_empty() {
        archive.unpack(staging).map_err(|e| format!("{url}: {e}"))?;
    } else {
        let mut nested_archives = Vec::new();
        for entry in archive.entries().map_err(|e| format!("{url}: {e}"))? {
            let mut entry = entry.map_err(|e| format!("{url}: {e}"))?;
            let path = entry
                .path()
                .map_err(|e| format!("{url}: {e}"))?
                .into_owned();
            // GNU tar packs `./name` as often as `name`; match,
            // unpack, and collect against the stripped path.
            let path = path.strip_prefix("./").unwrap_or(&path);
            if model.include.iter().any(|prefix| path.starts_with(prefix)) {
                entry
                    .unpack_in(staging)
                    .map_err(|e| format!("{url}: {}: {e}", path.display()))?;
                // String compare, not Path::ends_with: that compares
                // components and would not see the suffix.
                let name = path.to_string_lossy();
                if model.nested && TARBALL_SUFFIXES.iter().any(|s| name.ends_with(s)) {
                    nested_archives.push(staging.join(path));
                }
            }
        }
        for nested in nested_archives {
            let parent = nested
                .parent()
                .ok_or_else(|| format!("{}: no parent", nested.display()))?;
            let file = File::open(&nested).map_err(|e| format!("{}: {e}", nested.display()))?;
            tar::Archive::new(flate2::read::GzDecoder::new(BufReader::new(file)))
                .unpack(parent)
                .map_err(|e| format!("{}: {e}", nested.display()))?;
            fs::remove_file(&nested).map_err(|e| format!("{}: {e}", nested.display()))?;
        }
    }
    Ok(staging.join(&model.name))
}

fn fetch(model: &Model, dest: &Path) -> Result {
    let target = dest.join(&model.name);
    if target.exists() && verify(model, &target).is_ok() {
        eprintln!("{}: present and verified", model.id);
        return Ok(());
    }
    let staging = dest.join(format!(".tmp-{}", random_suffix()));
    fs::create_dir_all(&staging).map_err(|e| format!("{}: {e}", staging.display()))?;
    let result = (|| {
        let download_path = staging.join("download");
        download(&model.archive, &download_path)?;
        let unpacked = unpack(model, &download_path, &staging)?;
        for extra in &model.extras {
            download(&extra.download, &unpacked.join(&extra.path))?;
        }
        verify(model, &unpacked)?;
        if target.exists() {
            if target.is_dir() {
                fs::remove_dir_all(&target)
            } else {
                fs::remove_file(&target)
            }
            .map_err(|e| format!("{}: {e}", target.display()))?;
        }
        fs::rename(&unpacked, &target).map_err(|e| format!("{}: {e}", target.display()))
    })();
    let _ = fs::remove_dir_all(&staging);
    result?;
    eprintln!("{}: fetched and verified", model.id);
    Ok(())
}

/// Fetches every entry of `models`, or only those whose ID is in
/// `only`, into `dest` and prints the `env_prefix`-based variables
/// tests read. When `GITHUB_ENV` is set, they are appended there too.
fn fetch_all(
    models: &[Model],
    only: &[String],
    dest: &Path,
    env_prefix: &str,
    list: &str,
) -> Result {
    for id in only {
        if !models.iter().any(|m| &m.id == id) {
            return Err(format!("unknown {list} {id}"));
        }
    }
    fs::create_dir_all(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let mut exports = String::new();
    for model in models {
        if !only.is_empty() && !only.contains(&model.id) {
            continue;
        }
        fetch(model, dest)?;
        let line = format!(
            "{}={}\n",
            env_var(env_prefix, &model.id),
            dest.join(&model.name).display()
        );
        exports.push_str(&line);
    }
    print!("{exports}");
    if let Some(path) = std::env::var_os("GITHUB_ENV") {
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|e| e.to_string())?;
        file.write_all(exports.as_bytes())
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Fetches the models of `fixtures/manifest.json`.
pub(crate) fn fetch_fixtures(only: &[String]) -> Result {
    fetch_all(
        &load_manifest()?.models,
        only,
        &models_dir(),
        "SPEECHKIT_MODEL",
        "model",
    )
}

/// Fetches the evaluation datasets of `fixtures/evals.json`.
pub(crate) fn fetch_evals(only: &[String]) -> Result {
    fetch_all(
        &load_evals()?.evals,
        only,
        &evals_dir(),
        "SPEECHKIT_EVAL",
        "eval",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_parses_and_ids_are_unique() {
        let manifest = load_manifest().unwrap();
        check_invariants(&manifest.models);
    }

    #[test]
    fn evals_manifest_parses_and_ids_are_unique() {
        check_invariants(&load_evals().unwrap().evals);
    }

    /// The shape every manifest entry must have.
    fn check_invariants(models: &[Model]) {
        let mut ids: Vec<_> = models.iter().map(|m| m.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), models.len());
        for model in models {
            assert_eq!(model.archive.sha256.len(), 64, "{}", model.id);
            assert!(model.archive.url.starts_with("https://"), "{}", model.id);
            for prefix in &model.include {
                // A member path is relative, so an absolute or empty
                // prefix can never match one.
                assert!(
                    !prefix.is_empty() && !prefix.starts_with('/'),
                    "{}",
                    model.id
                );
            }
            for extra in &model.extras {
                assert_eq!(extra.download.sha256.len(), 64, "{}", model.id);
                assert!(
                    model.files.iter().any(|file| file.path == extra.path),
                    "{}: extra {} is not listed in files",
                    model.id,
                    extra.path
                );
            }
        }
    }

    #[test]
    fn env_var_names() {
        assert_eq!(
            env_var("SPEECHKIT_MODEL", "streaming-en"),
            "SPEECHKIT_MODEL_STREAMING_EN"
        );
        assert_eq!(
            env_var("SPEECHKIT_MODEL", "silero-vad"),
            "SPEECHKIT_MODEL_SILERO_VAD"
        );
        assert_eq!(
            env_var("SPEECHKIT_EVAL", "aishell-1"),
            "SPEECHKIT_EVAL_AISHELL_1"
        );
    }

    #[test]
    fn verify_checks_hashes() {
        let dir = std::env::temp_dir().join(format!("xtask-verify-{}", random_suffix()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "abc").unwrap();
        let model = Model {
            id: "x".into(),
            name: "x".into(),
            archive: Archive {
                url: String::new(),
                sha256: String::new(),
                size: 0,
            },
            files: vec![FileEntry {
                path: "a.txt".into(),
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            }],
            extras: Vec::new(),
            include: Vec::new(),
            nested: false,
        };
        verify(&model, &dir).unwrap();
        fs::write(dir.join("a.txt"), "abd").unwrap();
        assert!(verify(&model, &dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tar_gz_unpacks_and_include_filters() {
        let dir = std::env::temp_dir().join(format!("xtask-targz-{}", random_suffix()));
        fs::create_dir_all(&dir).unwrap();
        let archive_path = dir.join("eval.tar.gz");
        let encoder = flate2::write::GzEncoder::new(
            File::create(&archive_path).unwrap(),
            flate2::Compression::default(),
        );
        let mut tar = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        for (path, contents) in [("root/a.txt", "alpha"), ("root/sub/b.txt", "beta")] {
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, path, contents.as_bytes())
                .unwrap();
        }
        tar.into_inner().unwrap().finish().unwrap();
        let model = Model {
            id: "eval".into(),
            name: "root".into(),
            archive: Archive {
                url: "https://example.com/eval.tar.gz".into(),
                sha256: String::new(),
                size: 0,
            },
            files: Vec::new(),
            extras: Vec::new(),
            include: Vec::new(),
            nested: false,
        };
        let all = dir.join("all");
        fs::create_dir_all(&all).unwrap();
        unpack(&model, &archive_path, &all).unwrap();
        assert_eq!(fs::read_to_string(all.join("root/a.txt")).unwrap(), "alpha");
        assert_eq!(
            fs::read_to_string(all.join("root/sub/b.txt")).unwrap(),
            "beta"
        );
        let model = Model {
            include: vec!["root/sub".into()],
            ..model
        };
        let sub = dir.join("sub");
        fs::create_dir_all(&sub).unwrap();
        unpack(&model, &archive_path, &sub).unwrap();
        assert!(!sub.join("root/a.txt").exists());
        assert_eq!(
            fs::read_to_string(sub.join("root/sub/b.txt")).unwrap(),
            "beta"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn include_matches_dot_slash_members() {
        let dir = std::env::temp_dir().join(format!("xtask-dotslash-{}", random_suffix()));
        fs::create_dir_all(&dir).unwrap();
        let archive_path = dir.join("dotslash.tar.gz");
        let encoder = flate2::write::GzEncoder::new(
            File::create(&archive_path).unwrap(),
            flate2::Compression::default(),
        );
        let mut tar = tar::Builder::new(encoder);
        let contents = "alpha";
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o644);
        // tar-rs strips `.` components on write, so the name is set on
        // the raw header: this is what `tar czf x.tgz ./root` packs.
        let name = b"./root/a.txt";
        header.as_old_mut().name[..name.len()].copy_from_slice(name);
        header.set_cksum();
        tar.append(&header, contents.as_bytes()).unwrap();
        tar.into_inner().unwrap().finish().unwrap();
        let model = Model {
            id: "eval".into(),
            name: "root".into(),
            archive: Archive {
                url: "https://example.com/dotslash.tar.gz".into(),
                sha256: String::new(),
                size: 0,
            },
            files: Vec::new(),
            extras: Vec::new(),
            include: vec!["root/a.txt".into()],
            nested: false,
        };
        let staging = dir.join("staging");
        fs::create_dir_all(&staging).unwrap();
        unpack(&model, &archive_path, &staging).unwrap();
        assert_eq!(
            fs::read_to_string(staging.join("root/a.txt")).unwrap(),
            "alpha"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn nested_tar_gz_members_unpack_in_place() {
        let dir = std::env::temp_dir().join(format!("xtask-nested-{}", random_suffix()));
        fs::create_dir_all(&dir).unwrap();
        // An inner tarball holding root/sub/c.txt.
        let inner_path = dir.join("inner.tar.gz");
        let encoder = flate2::write::GzEncoder::new(
            File::create(&inner_path).unwrap(),
            flate2::Compression::default(),
        );
        let mut tar = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(5);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(&mut header, "sub/c.txt", &b"gamma"[..])
            .unwrap();
        tar.into_inner().unwrap().finish().unwrap();
        // An outer tarball whose only member is the inner one.
        let archive_path = dir.join("outer.tar.gz");
        let encoder = flate2::write::GzEncoder::new(
            File::create(&archive_path).unwrap(),
            flate2::Compression::default(),
        );
        let mut tar = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        let size = fs::metadata(&inner_path).unwrap().len();
        header.set_size(size);
        header.set_mode(0o644);
        header.set_cksum();
        tar.append_data(
            &mut header,
            "root/pkg.tar.gz",
            File::open(&inner_path).unwrap(),
        )
        .unwrap();
        tar.into_inner().unwrap().finish().unwrap();
        let model = Model {
            include: vec!["root/pkg.tar.gz".into()],
            nested: true,
            ..Model {
                id: "eval".into(),
                name: "root".into(),
                archive: Archive {
                    url: "https://example.com/outer.tar.gz".into(),
                    sha256: String::new(),
                    size: 0,
                },
                files: Vec::new(),
                extras: Vec::new(),
                include: Vec::new(),
                nested: false,
            }
        };
        let staging = dir.join("staging");
        fs::create_dir_all(&staging).unwrap();
        unpack(&model, &archive_path, &staging).unwrap();
        assert_eq!(
            fs::read_to_string(staging.join("root/sub/c.txt")).unwrap(),
            "gamma"
        );
        assert!(!staging.join("root/pkg.tar.gz").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
