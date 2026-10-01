//! `cargo xtask release-check`: everything that must hold before a
//! release.
//!
//! Every check runs, even after one fails, so one run lists everything to
//! fix. The file checks are pure functions and unit-tested; the others run
//! tools: cargo, cargo-deny, cargo-semver-checks, and the GitHub CLI. The
//! semver check asks crates.io which crates have been published before.

use std::{
    fs,
    process::Command,
    time::{Duration, SystemTime},
};

use serde::Deserialize;

use crate::{Result, workspace};

/// The MSRV, as in the workspace manifest.
const MSRV: &str = "1.88";

/// How recent the model run on `HEAD` or its parent must be.
const MODELS_MAX_AGE: Duration = Duration::from_secs(24 * 3600);

/// Runs a command in the workspace, succeeding if it exits with 0.
fn run(program: &str, args: &[&str]) -> Result {
    let status = Command::new(program)
        .args(args)
        .current_dir(workspace())
        .status()
        .map_err(|e| format!("could not run {program}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`{program} {}` failed", args.join(" ")))
    }
}

fn output(program: &str, args: &[&str]) -> Result<String> {
    let out = Command::new(program)
        .args(args)
        .current_dir(workspace())
        .output()
        .map_err(|e| format!("could not run {program}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`{program} {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    String::from_utf8(out.stdout).map_err(|e| e.to_string())
}

#[derive(Deserialize)]
struct Metadata {
    packages: Vec<Package>,
}

#[derive(Deserialize)]
struct Package {
    name: String,
    version: String,
    /// `None` means publishable anywhere; an empty list means `publish = false`.
    publish: Option<Vec<String>>,
}

impl Package {
    fn published(&self) -> bool {
        self.publish
            .as_ref()
            .is_none_or(|registries| !registries.is_empty())
    }
}

/// The workspace version from the root manifest's `[workspace.package]`.
pub(crate) fn workspace_version(manifest: &str) -> Result<String> {
    let mut in_section = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_section = line == "[workspace.package]";
        } else if in_section && let Some(value) = line.strip_prefix("version") {
            let value = value.trim_start().strip_prefix('=').map(str::trim);
            if let Some(version) = value.and_then(|v| v.strip_prefix('"')?.strip_suffix('"')) {
                return Ok(version.to_owned());
            }
        }
    }
    Err("no version in [workspace.package]".into())
}

/// Published crates whose version is not the workspace version.
fn mismatched_versions<'a>(packages: &'a [Package], version: &str) -> Vec<&'a str> {
    packages
        .iter()
        .filter(|p| p.published() && p.version != version)
        .map(|p| p.name.as_str())
        .collect()
}

/// Whether the changelog has a section for `version`, such as
/// `## [0.3.0]` or `## 0.3.0`.
pub(crate) fn changelog_has(changelog: &str, version: &str) -> bool {
    changelog.lines().any(|line| {
        let Some(title) = line.strip_prefix("## ") else {
            return false;
        };
        let title = title.trim_start_matches('[');
        title
            .strip_prefix(version)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with([']', ' ']))
    })
}

#[derive(Deserialize)]
struct Run {
    conclusion: Option<String>,
    #[serde(rename = "updatedAt")]
    updated_at: String,
}

/// Seconds since the Unix epoch for an RFC 3339 UTC time such as
/// `2026-09-24T03:12:45Z`.
pub(crate) fn parse_utc(time: &str) -> Option<u64> {
    let time = time.strip_suffix('Z')?;
    let (date, clock) = time.split_once('T')?;
    let mut date = date.split('-').map(str::parse::<i64>);
    let (year, month, day) = (date.next()?.ok()?, date.next()?.ok()?, date.next()?.ok()?);
    let clock = clock.split('.').next()?;
    let mut clock = clock.split(':').map(str::parse::<i64>);
    let (hour, minute, second) = (
        clock.next()?.ok()?,
        clock.next()?.ok()?,
        clock.next()?.ok()?,
    );
    // Days from the civil date (Howard Hinnant's algorithm).
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let month_index = (month + 9) % 12;
    let day_of_year = (153 * month_index + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    u64::try_from(days * 86_400 + hour * 3_600 + minute * 60 + second).ok()
}

/// Whether `models.yml` passed in the last day on `HEAD`, or on its parent
/// when `HEAD` is a release commit that changed only the version and the
/// changelog.
fn models_passed_recently() -> Result {
    let mut commits = Vec::new();
    for rev in ["HEAD", "HEAD^"] {
        let commit = output("git", &["rev-parse", rev])?;
        let commit = commit.trim().to_owned();
        if models_passed_on(&commit)? {
            return Ok(());
        }
        commits.push(commit[..commit.len().min(12)].to_owned());
    }
    Err(format!(
        "no successful models.yml run on {} in the last 24 hours",
        commits.join(" or ")
    ))
}

/// Whether `models.yml` passed on `commit` in the last day.
fn models_passed_on(commit: &str) -> Result<bool> {
    let json = output(
        "gh",
        &[
            "run",
            "list",
            "--workflow",
            "models.yml",
            "--commit",
            commit,
            "--json",
            "conclusion,updatedAt",
        ],
    )?;
    let runs: Vec<Run> = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    Ok(runs.iter().any(|run| {
        run.conclusion.as_deref() == Some("success")
            && parse_utc(&run.updated_at)
                .is_some_and(|at| now.saturating_sub(at) <= MODELS_MAX_AGE.as_secs())
    }))
}

fn read(path: &str) -> Result<String> {
    fs::read_to_string(workspace().join(path)).map_err(|e| format!("{path}: {e}"))
}

/// The checks that run tools.
fn tool_checks(published: &[&str]) -> Vec<(&'static str, Result)> {
    let clippy = [
        "clippy",
        "--workspace",
        "--all-targets",
        "--all-features",
        "--",
        "-D",
        "warnings",
    ];
    let msrv = format!("+{MSRV}");
    vec![
        ("fmt", run("cargo", &["fmt", "--all", "--", "--check"])),
        ("clippy", run("cargo", &clippy)),
        (
            "docs",
            run(
                "cargo",
                &["doc", "--workspace", "--all-features", "--no-deps"],
            ),
        ),
        ("deny", run("cargo", &["deny", "check"])),
        (
            "msrv",
            run("cargo", &[&msrv, "check", "--workspace", "--all-features"]),
        ),
        ("semver", semver(published)),
        ("models", models_passed_recently()),
    ]
}

/// Whether `name` has a version on crates.io.
fn on_crates_io(name: &str) -> Result<bool> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("speechkit-xtask (https://github.com/speechkit/speechkit)")
        .build()
        .map_err(|e| e.to_string())?;
    let response = client
        .get(format!("https://crates.io/api/v1/crates/{name}"))
        .send()
        .map_err(|e| format!("crates.io: {e}"))?;
    match response.status() {
        status if status.is_success() => Ok(true),
        reqwest::StatusCode::NOT_FOUND => Ok(false),
        status => Err(format!("crates.io answered {status} for {name}")),
    }
}

/// Published crates whose Rust API is not covered by semver. The CLI's
/// library exists so its tests can run commands in-process; every new flag
/// adds a field to a public args struct.
const NO_SEMVER: &[&str] = &["speechkit-cli"];

/// Runs cargo-semver-checks on every published crate that is already on
/// crates.io, except those in [`NO_SEMVER`]. A crate's first release has
/// nothing to compare against, so it is skipped and named.
fn semver(published: &[&str]) -> Result {
    let mut first = Vec::new();
    for name in published {
        if NO_SEMVER.contains(name) {
            eprintln!("semver: {name} is not covered by semver; skipped");
        } else if on_crates_io(name)? {
            run("cargo", &["semver-checks", "check-release", "-p", name])?;
        } else {
            first.push(*name);
        }
    }
    if !first.is_empty() {
        eprintln!(
            "semver: first release of {}; nothing to compare",
            first.join(", ")
        );
    }
    Ok(())
}

/// The checks of files in the repository.
fn file_checks(packages: &[Package], version: &str) -> Vec<(&'static str, Result)> {
    let changelog = read("CHANGELOG.md").and_then(|text| {
        if changelog_has(&text, version) {
            Ok(())
        } else {
            Err(format!("CHANGELOG.md has no section for {version}"))
        }
    });
    let mismatched = mismatched_versions(packages, version);
    let versions = if mismatched.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "not at the workspace version {version}: {}",
            mismatched.join(", ")
        ))
    };
    vec![("changelog", changelog), ("versions", versions)]
}

/// Runs every check and reports each result.
pub(crate) fn release_check() -> Result {
    let metadata: Metadata = serde_json::from_str(&output(
        "cargo",
        &["metadata", "--format-version", "1", "--no-deps"],
    )?)
    .map_err(|e| e.to_string())?;
    let version = workspace_version(&read("Cargo.toml")?)?;
    let published: Vec<&str> = metadata
        .packages
        .iter()
        .filter(|p| p.published())
        .map(|p| p.name.as_str())
        .collect();
    let mut checks = tool_checks(&published);
    checks.extend(file_checks(&metadata.packages, &version));
    let mut failed = 0;
    for (name, result) in &checks {
        match result {
            Ok(()) => eprintln!("ok      {name}"),
            Err(error) => {
                failed += 1;
                eprintln!("FAILED  {name}: {error}");
            }
        }
    }
    if failed == 0 {
        eprintln!("ready to release {version}");
        Ok(())
    } else {
        Err(format!("{failed} of {} checks failed", checks.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_workspace_version() {
        let manifest = "[workspace]\nmembers = []\n\n[workspace.package]\nedition = \"2024\"\nversion = \"0.3.0\"\n\n[workspace.dependencies]\nversion = \"9\"\n";
        assert_eq!(workspace_version(manifest).unwrap(), "0.3.0");
        assert!(workspace_version("[package]\nversion = \"1\"\n").is_err());
    }

    #[test]
    fn finds_changelog_sections() {
        let changelog = "# Changelog\n\n## [0.3.0] - 2026-10-01\n\n## 0.2.0\n\n## [0.1.0-rc.1]\n";
        assert!(changelog_has(changelog, "0.3.0"));
        assert!(changelog_has(changelog, "0.2.0"));
        assert!(!changelog_has(changelog, "0.1.0"));
        assert!(changelog_has(changelog, "0.1.0-rc.1"));
        assert!(!changelog_has(changelog, "0.3"));
    }

    #[test]
    fn checks_versions_of_published_crates_only() {
        let package = |name: &str, version: &str, publish: Option<Vec<String>>| Package {
            name: name.into(),
            version: version.into(),
            publish,
        };
        let packages = [
            package("a", "0.3.0", None),
            package("b", "0.2.0", None),
            package("xtask", "0.0.1", Some(Vec::new())),
        ];
        assert_eq!(mismatched_versions(&packages, "0.3.0"), ["b"]);
    }

    #[test]
    fn parses_utc_times() {
        assert_eq!(parse_utc("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_utc("2000-03-01T00:00:00Z"), Some(951_868_800));
        assert_eq!(parse_utc("2026-09-24T03:12:45.123Z"), Some(1_790_219_565));
        assert_eq!(parse_utc("2026-09-24 03:12:45"), None);
    }
}
