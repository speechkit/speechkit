//! Switches for expensive tests.

/// Whether large-model tests (Qwen3-ASR, FunASR-Nano, FireRed) should
/// run, given the value of `SPEECHKIT_LARGE_MODEL_TEST`: only `"1"` or
/// `"true"` enable them.
///
/// Call it at the top of each large-model test body, not in a `cfg`, so
/// the tests always compile:
///
/// ```
/// # fn test() {
/// let gate = std::env::var("SPEECHKIT_LARGE_MODEL_TEST").ok();
/// if !speechkit_testkit::gates::large_model_tier_from(gate.as_deref()) {
///     return;
/// }
/// # }
/// ```
pub fn large_model_tier_from(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "true"))
}

/// [`large_model_tier_from`] applied to the environment, with a note on
/// stderr when the tier is off.
pub fn large_model_tier() -> bool {
    let enabled =
        large_model_tier_from(std::env::var("SPEECHKIT_LARGE_MODEL_TEST").ok().as_deref());
    if !enabled {
        eprintln!("skipped: set SPEECHKIT_LARGE_MODEL_TEST=1 to run large-model tests");
    }
    enabled
}

/// The directory of model `id`, from `SPEECHKIT_MODEL_<ID>` with `-`
/// read as `_`, or `None` with a note on stderr when it is not set.
pub fn model_dir(id: &str) -> Option<std::path::PathBuf> {
    let var = format!("SPEECHKIT_MODEL_{}", id.to_uppercase().replace('-', "_"));
    let path = std::env::var_os(&var).map(std::path::PathBuf::from);
    if path.is_none() {
        eprintln!("skipped: {var} is not set; run `cargo xtask fetch-fixtures`");
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_one_and_true_enable() {
        assert!(large_model_tier_from(Some("1")));
        assert!(large_model_tier_from(Some("true")));
        for off in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("TRUE"),
            Some("yes"),
            Some(" 1"),
        ] {
            assert!(!large_model_tier_from(off), "{off:?}");
        }
    }
}
