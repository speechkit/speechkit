//! Secrets such as API keys.

use std::fmt;

use zeroize::Zeroizing;

use super::SpeechError;

/// A secret value, such as an API key.
///
/// - `Debug` and `Display` print `Secret(***)`.
/// - The memory is zeroed when the secret is dropped.
/// - It is not `Clone`: share it through `Arc<Secret>`.
/// - It never serializes. With the `serde` feature it can be deserialized.
/// - [`expose`](Self::expose) is the only way to read it.
pub struct Secret(Zeroizing<String>);

impl Secret {
    /// Wraps `value`.
    pub fn new(value: impl Into<String>) -> Self {
        Self(Zeroizing::new(value.into()))
    }

    /// Reads the secret from the environment variable `var`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] if the variable is missing, empty, or
    /// not valid Unicode. The message names the variable, never its value.
    pub fn from_env(var: &str) -> Result<Self, SpeechError> {
        match std::env::var(var) {
            Ok(value) if !value.trim().is_empty() => Ok(Self::new(value)),
            Ok(_) => Err(SpeechError::InvalidInput(format!(
                "environment variable {var} is empty"
            ))),
            Err(_) => Err(SpeechError::InvalidInput(format!(
                "environment variable {var} is not set"
            ))),
        }
    }

    /// The secret value.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(***)")
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(***)")
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Secret {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::new)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[derive(Debug)]
    #[expect(dead_code, reason = "fields are read only through Debug")]
    struct Config {
        endpoint: String,
        api_key: Arc<Secret>,
    }

    #[test]
    fn secret_never_formats() {
        let config = Config {
            endpoint: "https://example.com".into(),
            api_key: Arc::new(Secret::new("sk-very-secret")),
        };
        for text in [
            format!("{config:?}"),
            format!("{config:#?}"),
            format!("{}", config.api_key),
        ] {
            assert!(!text.contains("sk-very-secret"), "{text}");
            assert!(text.contains("Secret(***)"), "{text}");
        }
        assert_eq!(config.api_key.expose(), "sk-very-secret");
    }

    #[test]
    fn missing_env_var_names_the_variable() {
        let error = Secret::from_env("SPEECHKIT_TEST_SURELY_UNSET_VARIABLE").unwrap_err();
        assert_eq!(
            error.to_string(),
            "invalid input: environment variable SPEECHKIT_TEST_SURELY_UNSET_VARIABLE is not set"
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserializes_from_a_string() {
        let secret: Secret = serde_json::from_str("\"abc\"").unwrap();
        assert_eq!(secret.expose(), "abc");
    }
}
