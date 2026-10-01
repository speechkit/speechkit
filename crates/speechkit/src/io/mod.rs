#![doc = include_str!("README.md")]

mod capture;
mod convert;
mod speaker;

#[doc(hidden)]
pub use capture::FakeMicrophone;
pub use capture::{
    Capture, CaptureOptions, ListenOptions, Listening, Microphone, Recording, Wake, WakeUpdate,
    WatchOptions, Watching,
};
#[doc(hidden)]
pub use speaker::FakeSpeaker;
pub use speaker::{Playback, Sink, Sound, Speaker};

/// An audio device, as [`Microphone::list`] or [`Speaker::list`] lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeviceInfo {
    /// The name to pass to `Microphone::open` or `Speaker::open`.
    pub name: String,
    /// Whether this is the system's default device.
    pub is_default: bool,
}

/// What the errors for an input or output device say.
fn direction(input: bool) -> (&'static str, &'static str) {
    if input {
        ("microphone", "input")
    } else {
        ("speaker", "output")
    }
}

/// The input or output devices with their names. Devices that no longer
/// answer are left out.
fn named_devices(input: bool) -> Result<Vec<(cpal::Device, String)>, crate::SpeechError> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let (backend, _) = direction(input);
    let host = cpal::default_host();
    let devices: Vec<cpal::Device> = if input {
        host.input_devices().map(Iterator::collect)
    } else {
        host.output_devices().map(Iterator::collect)
    }
    .map_err(|error| crate::SpeechError::backend(backend, true, error.to_string()))?;
    Ok(devices
        .into_iter()
        .filter_map(|device| {
            let name = device.description().ok()?.name().to_owned();
            Some((device, name))
        })
        .collect())
}

fn list(input: bool) -> Result<Vec<DeviceInfo>, crate::SpeechError> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let default = if input {
        host.default_input_device()
    } else {
        host.default_output_device()
    }
    .and_then(|device| device.id().ok());
    Ok(named_devices(input)?
        .into_iter()
        .map(|(device, name)| DeviceInfo {
            is_default: default.is_some() && device.id().ok() == default,
            name,
        })
        .collect())
}

/// The index of the device called `wanted`, ignoring spaces around it.
/// Matches are tried in order: the exact name, the name ignoring case, then a part of the name
/// ignoring case. The first kind that matches must match only one device.
///
/// # Errors
///
/// [`SpeechError::InvalidInput`](crate::SpeechError::InvalidInput) if
/// `wanted` is empty or all whitespace, or naming the candidates if no
/// name matches or the first kind that matches matches more than one.
fn pick(names: &[String], wanted: &str, kind: &str) -> Result<usize, crate::SpeechError> {
    use crate::SpeechError;

    let wanted = wanted.trim();
    if wanted.is_empty() {
        return Err(SpeechError::InvalidInput(format!(
            "the {kind} device name is empty"
        )));
    }
    let needle = wanted.to_lowercase();
    let folded: Vec<String> = names.iter().map(|name| name.to_lowercase()).collect();
    let tiers: [&dyn Fn(usize) -> bool; 3] = [
        &|index| names[index] == wanted,
        &|index| folded[index] == needle,
        &|index| folded[index].contains(&needle),
    ];
    for matches_name in tiers {
        let matches: Vec<usize> = (0..names.len())
            .filter(|&index| matches_name(index))
            .collect();
        match matches.as_slice() {
            [] => {}
            [index] => return Ok(*index),
            [first, rest @ ..] if rest.iter().all(|&index| names[index] == names[*first]) => {
                return Err(SpeechError::InvalidInput(format!(
                    "{} {kind} devices are called {:?}, and speechkit cannot tell them apart",
                    matches.len(),
                    names[*first]
                )));
            }
            _ => {
                return Err(SpeechError::InvalidInput(format!(
                    "{wanted:?} matches several {kind} devices: {}",
                    quoted(matches.iter().map(|&index| &names[index]))
                )));
            }
        }
    }
    Err(SpeechError::InvalidInput(format!(
        "no {kind} device matches {wanted:?}; available: {}",
        quoted(names.iter())
    )))
}

fn quoted<'a>(names: impl Iterator<Item = &'a String>) -> String {
    let names: Vec<String> = names.map(|name| format!("{name:?}")).collect();
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
    }
}

/// The input device called `name` if `input`, otherwise the output device,
/// or the default one if `name` is `None`, with its name, its default
/// configuration, and its rate. See [`pick`] for how a name matches.
///
/// # Errors
///
/// `InvalidInput` if `name` is empty or matches no device or several, a
/// retryable backend error if there is no default device or it has no
/// usable configuration, or `Unsupported` for its sample format.
fn open(input: bool, name: Option<&str>) -> Result<Opened, crate::SpeechError> {
    use crate::SpeechError;
    use cpal::traits::{DeviceTrait, HostTrait};

    let (backend, kind) = direction(input);
    let failed = |error: String| SpeechError::backend(backend, true, error);
    let (device, name) = if let Some(wanted) = name {
        let (devices, names): (Vec<_>, Vec<_>) = named_devices(input)?.into_iter().unzip();
        let index = pick(&names, wanted, kind)?;
        let name = names[index].clone();
        let device = devices
            .into_iter()
            .nth(index)
            .expect("pick returns an index into names, which has one entry per device");
        (device, name)
    } else {
        let host = cpal::default_host();
        if input {
            host.default_input_device()
        } else {
            host.default_output_device()
        }
        .ok_or_else(|| failed(format!("no {kind} device")))
        .map(|device| {
            let name = device
                .description()
                .map(|description| description.name().to_owned())
                .unwrap_or_default();
            (device, name)
        })?
    };
    let config = if input {
        device.default_input_config()
    } else {
        device.default_output_config()
    }
    .map_err(|error| failed(error.to_string()))?;
    if !matches!(
        config.sample_format(),
        cpal::SampleFormat::F32 | cpal::SampleFormat::I16 | cpal::SampleFormat::U16
    ) {
        return Err(SpeechError::Unsupported(format!(
            "the {kind} device uses {:?}; only f32, i16, and u16 are supported",
            config.sample_format()
        )));
    }
    let rate = crate::SampleRate::new(config.sample_rate())?;
    Ok(Opened {
        device,
        name,
        config,
        rate,
    })
}

/// A device [`open`] found.
struct Opened {
    device: cpal::Device,
    name: String,
    config: cpal::SupportedStreamConfig,
    rate: crate::SampleRate,
}

#[cfg(test)]
mod tests {
    use super::pick;
    use crate::SpeechError;

    fn names() -> Vec<String> {
        ["MacBook Pro Microphone", "BlackHole 2ch", "BlackHole 16ch"]
            .map(String::from)
            .to_vec()
    }

    #[test]
    fn an_exact_name_wins() {
        assert_eq!(pick(&names(), "BlackHole 2ch", "input").unwrap(), 1);
    }

    #[test]
    fn a_unique_part_of_a_name_matches_ignoring_case() {
        assert_eq!(pick(&names(), "macbook", "input").unwrap(), 0);
        assert_eq!(pick(&names(), "16CH", "input").unwrap(), 2);
    }

    #[test]
    fn an_exact_name_wins_over_a_longer_one_containing_it() {
        let names = vec!["USB".to_owned(), "USB Audio".to_owned()];
        assert_eq!(pick(&names, "USB", "input").unwrap(), 0);
    }

    #[test]
    fn a_whole_name_ignoring_case_wins_over_a_longer_one_containing_it() {
        let names = vec!["USB".to_owned(), "USB Audio".to_owned()];
        assert_eq!(pick(&names, "usb", "input").unwrap(), 0);
    }

    #[test]
    fn duplicate_names_are_an_error() {
        let names = vec![
            "USB Audio".to_owned(),
            "MacBook Pro Microphone".to_owned(),
            "USB Audio".to_owned(),
        ];
        for wanted in ["USB Audio", "usb audio", "usb"] {
            let error = pick(&names, wanted, "input").unwrap_err();
            assert_eq!(
                error.to_string(),
                "invalid input: 2 input devices are called \"USB Audio\", \
                 and speechkit cannot tell them apart",
                "{wanted}"
            );
        }
    }

    #[test]
    fn spaces_around_a_name_are_ignored() {
        assert_eq!(pick(&names(), " BlackHole 2ch\t", "input").unwrap(), 1);
        assert_eq!(pick(&names(), "  16ch ", "input").unwrap(), 2);
    }

    #[test]
    fn an_empty_name_is_an_error() {
        for wanted in ["", "  ", "\t"] {
            let error = pick(&names(), wanted, "input").unwrap_err();
            assert!(matches!(error, SpeechError::InvalidInput(_)), "{error:?}");
            assert!(error.to_string().contains("name is empty"), "{error}");
        }
    }

    #[test]
    fn several_matches_are_named() {
        let error = pick(&names(), "blackhole", "input").unwrap_err();
        let SpeechError::InvalidInput(message) = error else {
            panic!("expected InvalidInput, got {error:?}");
        };
        assert!(
            message.contains("\"BlackHole 2ch\", \"BlackHole 16ch\""),
            "{message}"
        );
    }

    #[test]
    fn no_match_lists_the_devices() {
        let error = pick(&names(), "headset", "output").unwrap_err();
        let SpeechError::InvalidInput(message) = error else {
            panic!("expected InvalidInput, got {error:?}");
        };
        assert!(
            message.starts_with("no output device matches \"headset\""),
            "{message}"
        );
        assert!(message.contains("\"MacBook Pro Microphone\""), "{message}");
    }

    #[test]
    fn no_devices_says_none() {
        let error = pick(&[], "x", "input").unwrap_err();
        assert!(error.to_string().contains("available: none"), "{error}");
    }
}
