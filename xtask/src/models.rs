//! `cargo xtask fetch-fixtures`: download and verify the models listed in
//! `fixtures/manifest.json`. `cargo xtask fetch-evals` does the same for
//! the evaluation datasets in `fixtures/evals.json`, which are fetched
//! only on request because they are far larger than the models, into a
//! cache of their own so corpora never sit among the models.
//!
//! Each tar archive is hashed as it streams in and unpacked on the fly
//! into `<dest>/.tmp-<random>`, never stored whole; once the last byte
//! has passed, its hash is checked, the result is checked file by file,
//! and then renamed into place, so a model directory is either complete
//! and verified or absent. A model's `extras` (files published
//! separately, such as a Matcha vocoder) are downloaded into its
//! directory before it is checked. An entry with `include` prefixes
//! unpacks only the archive members under one of them, so an evaluation
//! set can carry a corpus's test split without its training data; with
//! `nested`, members that are themselves tarballs (AISHELL-1 ships one
//! per speaker) are unpacked straight from the stream instead of being
//! written. Such an entry leaves a stamp of its filter in its directory,
//! so a changed `include` fetches again.

use std::{
    fs::{self, File},
    io::{self, BufReader, Read, Write},
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
    /// Whether members that are themselves `.tar.gz` or `.tgz`
    /// archives are unpacked where they sit instead of being written.
    /// Corpora such as AISHELL-1 ship one tar per speaker. With
    /// `include`, only the members it lets through are considered.
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

/// The file in a fetched directory that records an `include` or
/// `nested` filter; see [`stamp`].
const STAMP: &str = ".fetched";

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

/// A reader that hashes and counts every byte read through it.
struct Hashing<R> {
    inner: R,
    hasher: Sha256,
    size: u64,
}

impl<R: Read> Read for Hashing<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buf)?;
        self.hasher.update(&buf[..read]);
        self.size += read as u64;
        Ok(read)
    }
}

impl<R: Read> Hashing<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
            size: 0,
        }
    }

    /// Reads what is left of the stream, such as padding after a tar's
    /// end, and checks the whole against `archive`.
    fn finish(mut self, archive: &Archive) -> Result {
        io::copy(&mut self, &mut io::sink()).map_err(|e| format!("{}: {e}", archive.url))?;
        let hash = hex(&self.hasher.finalize());
        if self.size != archive.size || hash != archive.sha256 {
            return Err(format!(
                "{}: got {} bytes with sha256 {hash}, expected {} bytes with {}",
                archive.url, self.size, archive.size, archive.sha256
            ));
        }
        Ok(())
    }
}

/// Starts downloading `archive`, hashing it as it arrives.
fn open(archive: &Archive) -> Result<Hashing<reqwest::blocking::Response>> {
    eprintln!(
        "downloading {} ({} MB)",
        archive.url,
        archive.size / 1_000_000
    );
    let client = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .map_err(|e| e.to_string())?;
    let response = client
        .get(&archive.url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|e| format!("{}: {e}", archive.url))?;
    Ok(Hashing::new(response))
}

/// Streams `archive` into `out` and checks its hash.
fn download(archive: &Archive, out: &Path) -> Result {
    let mut source = open(archive)?;
    let mut file = File::create(out).map_err(|e| format!("{}: {e}", out.display()))?;
    io::copy(&mut source, &mut file)
        .map_err(|e| format!("{} -> {}: {e}", archive.url, out.display()))?;
    source.finish(archive)
}

fn random_suffix() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    format!("{}-{nanos}", std::process::id())
}

/// The decompressed tar stream of `raw` when `url` names a tar
/// archive, or `None` for a plain file.
fn tar_stream<'a>(url: &str, raw: impl Read + 'a) -> Option<Box<dyn Read + 'a>> {
    if url.ends_with(".tar.bz2") {
        Some(Box::new(bzip2::read::BzDecoder::new(BufReader::new(raw))))
    } else if TARBALL_SUFFIXES.iter().any(|s| url.ends_with(s)) {
        Some(Box::new(flate2::read::GzDecoder::new(BufReader::new(raw))))
    } else {
        None
    }
}

/// Unpacks the tar stream `tar` of `model` into `staging`, keeping
/// only members under an `include` prefix and unpacking `nested`
/// tarballs where they sit.
fn unpack(model: &Model, tar: impl Read, staging: &Path) -> Result {
    let url = &model.archive.url;
    let mut archive = tar::Archive::new(tar);
    if model.include.is_empty() && !model.nested {
        return archive.unpack(staging).map_err(|e| format!("{url}: {e}"));
    }
    for entry in archive.entries().map_err(|e| format!("{url}: {e}"))? {
        let mut entry = entry.map_err(|e| format!("{url}: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("{url}: {e}"))?
            .into_owned();
        // GNU tar packs `./name` as often as `name`; match and unpack
        // against the stripped path.
        let path = path.strip_prefix("./").unwrap_or(&path);
        if !model.include.is_empty() && !model.include.iter().any(|p| path.starts_with(p)) {
            continue;
        }
        // String compare, not Path::ends_with: that compares
        // components and would not see the suffix.
        let name = path.to_string_lossy();
        if model.nested && TARBALL_SUFFIXES.iter().any(|s| name.ends_with(s)) {
            let parent = path
                .parent()
                .map_or_else(|| staging.to_path_buf(), |p| staging.join(p));
            fs::create_dir_all(&parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            tar::Archive::new(flate2::read::GzDecoder::new(&mut entry))
                .unpack(&parent)
                .map_err(|e| format!("{url}: {name}: {e}"))?;
        } else {
            entry
                .unpack_in(staging)
                .map_err(|e| format!("{url}: {name}: {e}"))?;
        }
    }
    Ok(())
}

/// Downloads `model`'s archive into `staging` and returns the unpacked
/// root: the archive's top-level directory for a tar, which is
/// unpacked as it streams in, or the file itself for anything else.
/// The hash is checked after the last byte, so a bad tar is unpacked
/// before it is rejected; `fetch` then discards `staging`.
fn fetch_archive(model: &Model, staging: &Path) -> Result<PathBuf> {
    let archive = &model.archive;
    let mut source = open(archive)?;
    let Some(tar) = tar_stream(&archive.url, &mut source) else {
        let path = staging.join("download");
        let mut file = File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        io::copy(&mut source, &mut file)
            .map_err(|e| format!("{} -> {}: {e}", archive.url, path.display()))?;
        source.finish(archive)?;
        return Ok(path);
    };
    unpack(model, tar, staging)?;
    source.finish(archive)?;
    Ok(staging.join(&model.name))
}

/// What an `include` or `nested` entry unpacked, which its `files`
/// cannot vouch for: the filtered members are not listed there. It is
/// written to [`STAMP`] in the fetched directory, so a changed filter
/// fetches again. `None` for an entry that unpacks its whole archive.
fn stamp(model: &Model) -> Option<String> {
    (!model.include.is_empty() || model.nested).then(|| {
        format!(
            "{}\nnested={}\n{}\n",
            model.archive.sha256,
            model.nested,
            model.include.join("\n")
        )
    })
}

/// Whether `target` holds a complete fetch of `model`.
fn present(model: &Model, target: &Path) -> bool {
    target.exists()
        && verify(model, target).is_ok()
        && stamp(model).is_none_or(|stamp| {
            fs::read_to_string(target.join(STAMP)).is_ok_and(|found| found == stamp)
        })
}

fn fetch(model: &Model, dest: &Path) -> Result {
    let target = dest.join(&model.name);
    if present(model, &target) {
        eprintln!("{}: present and verified", model.id);
        return Ok(());
    }
    let staging = dest.join(format!(".tmp-{}", random_suffix()));
    fs::create_dir_all(&staging).map_err(|e| format!("{}: {e}", staging.display()))?;
    let result = (|| {
        let unpacked = fetch_archive(model, &staging)?;
        for extra in &model.extras {
            download(&extra.download, &unpacked.join(&extra.path))?;
        }
        verify(model, &unpacked)?;
        if let Some(stamp) = stamp(model) {
            let path = unpacked.join(STAMP);
            fs::write(&path, stamp).map_err(|e| format!("{}: {e}", path.display()))?;
        }
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

    /// Unpacks the tar file `archive` as [`fetch_archive`] would.
    fn unpack_file(model: &Model, archive: &Path, staging: &Path) {
        let tar = tar_stream(&model.archive.url, File::open(archive).unwrap()).unwrap();
        unpack(model, tar, staging).unwrap();
    }

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
        unpack_file(&model, &archive_path, &all);
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
        unpack_file(&model, &archive_path, &sub);
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
        unpack_file(&model, &archive_path, &staging);
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
        // With and without `include`: `nested` works on its own.
        let mut model = model;
        for (n, include) in [model.include.clone(), Vec::new()].into_iter().enumerate() {
            model.include = include;
            let staging = dir.join(format!("staging-{n}"));
            fs::create_dir_all(&staging).unwrap();
            unpack_file(&model, &archive_path, &staging);
            assert_eq!(
                fs::read_to_string(staging.join("root/sub/c.txt")).unwrap(),
                "gamma"
            );
            assert!(!staging.join("root/pkg.tar.gz").exists());
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hashing_checks_size_and_hash_after_the_last_byte() {
        let archive = Archive {
            url: "https://example.com/abc".into(),
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            size: 3,
        };
        let mut reader = Hashing::new(&b"abc"[..]);
        let mut first = [0_u8; 1];
        reader.read_exact(&mut first).unwrap();
        // The unread rest still counts.
        reader.finish(&archive).unwrap();
        assert!(Hashing::new(&b"abd"[..]).finish(&archive).is_err());
        assert!(Hashing::new(&b"abcd"[..]).finish(&archive).is_err());
    }

    #[test]
    fn a_changed_include_is_not_present() {
        let dir = std::env::temp_dir().join(format!("xtask-stamp-{}", random_suffix()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.txt"), "abc").unwrap();
        let model = Model {
            id: "eval".into(),
            name: "root".into(),
            archive: Archive {
                url: "https://example.com/eval.tar.gz".into(),
                sha256: String::new(),
                size: 0,
            },
            files: vec![FileEntry {
                path: "a.txt".into(),
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            }],
            extras: Vec::new(),
            include: vec!["root/a".into()],
            nested: false,
        };
        // Files verify, but nothing records what was unpacked.
        assert!(!present(&model, &dir));
        fs::write(dir.join(STAMP), stamp(&model).unwrap()).unwrap();
        assert!(present(&model, &dir));
        let widened = Model {
            include: vec!["root/a".into(), "root/b".into()],
            ..model
        };
        assert!(!present(&widened, &dir));
        // An entry without a filter needs no stamp.
        let whole = Model {
            include: Vec::new(),
            ..widened
        };
        fs::remove_file(dir.join(STAMP)).unwrap();
        assert!(present(&whole, &dir));
        fs::remove_dir_all(&dir).unwrap();
    }
}
