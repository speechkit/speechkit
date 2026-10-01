//! `cargo xtask fetch-fixtures`: download and verify the models listed in
//! `fixtures/manifest.json`.
//!
//! Each archive is streamed to disk while it is hashed, extracted into
//! `<dest>/.tmp-<random>`, checked file by file, and then renamed into
//! place, so a model directory is either complete and verified or absent.
//! A model's `extras` (files published separately, such as a Matcha
//! vocoder) are downloaded into its directory before it is checked.

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
struct Model {
    id: String,
    name: String,
    archive: Archive,
    files: Vec<FileEntry>,
    #[serde(default)]
    extras: Vec<Extra>,
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

/// `$SPEECHKIT_MODELS`, or `~/.cache/speechkit/models`.
pub(crate) fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("SPEECHKIT_MODELS") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".cache/speechkit/models")
}

/// The environment variable a test reads for model `id`.
pub(crate) fn env_var(id: &str) -> String {
    format!("SPEECHKIT_MODEL_{}", id.to_uppercase().replace('-', "_"))
}

fn load_manifest() -> Result<Manifest> {
    let path = workspace().join("fixtures/manifest.json");
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
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
        let unpacked = if model.archive.url.ends_with(".tar.bz2") {
            let file = File::open(&download_path).map_err(|e| e.to_string())?;
            tar::Archive::new(bzip2::read::BzDecoder::new(BufReader::new(file)))
                .unpack(&staging)
                .map_err(|e| format!("{}: {e}", model.archive.url))?;
            staging.join(&model.name)
        } else {
            download_path
        };
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

/// Fetches every model, or only those whose ID is in `only`, and prints
/// the environment variables tests read. When `GITHUB_ENV` is set, they
/// are appended there too.
pub(crate) fn fetch_fixtures(only: &[String]) -> Result {
    let manifest = load_manifest()?;
    for id in only {
        if !manifest.models.iter().any(|m| &m.id == id) {
            return Err(format!("unknown model {id}"));
        }
    }
    let dest = models_dir();
    fs::create_dir_all(&dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let mut exports = String::new();
    for model in &manifest.models {
        if !only.is_empty() && !only.contains(&model.id) {
            continue;
        }
        fetch(model, &dest)?;
        let line = format!(
            "{}={}\n",
            env_var(&model.id),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_parses_and_ids_are_unique() {
        let manifest = load_manifest().unwrap();
        let mut ids: Vec<_> = manifest.models.iter().map(|m| m.id.as_str()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), manifest.models.len());
        for model in &manifest.models {
            assert_eq!(model.archive.sha256.len(), 64, "{}", model.id);
            assert!(model.archive.url.starts_with("https://"), "{}", model.id);
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
        assert_eq!(env_var("streaming-en"), "SPEECHKIT_MODEL_STREAMING_EN");
        assert_eq!(env_var("silero-vad"), "SPEECHKIT_MODEL_SILERO_VAD");
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
        };
        verify(&model, &dir).unwrap();
        fs::write(dir.join("a.txt"), "abd").unwrap();
        assert!(verify(&model, &dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
