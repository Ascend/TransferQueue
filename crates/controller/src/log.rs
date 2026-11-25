// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Arguments;
use std::io::{self, Write};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Level {
    Debug = 10,
    Info = 20,
    Warning = 30,
    Error = 40,
}

impl Level {
    fn label(self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Error => "ERROR",
        }
    }
}

fn configured_level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("TQ_LOGGING_LEVEL") {
        Ok(raw) => match raw.trim().to_ascii_uppercase().as_str() {
            "DEBUG" => Level::Debug,
            "INFO" => Level::Info,
            "WARN" | "WARNING" => Level::Warning,
            "ERROR" | "CRITICAL" | "FATAL" => Level::Error,
            _ => Level::Warning,
        },
        Err(_) => Level::Warning,
    })
}

pub(crate) fn emit(level: Level, controller_id: &str, args: Arguments<'_>) {
    if level < configured_level() {
        return;
    }
    let timestamp = format_utc_timestamp(SystemTime::now());
    let mut stderr = io::stderr().lock();
    let _ = writeln!(
        stderr,
        "{timestamp} - {} - transfer_queue.rust_controller - [{controller_id}]: {args}",
        level.label()
    );
}

fn format_utc_timestamp(now: SystemTime) -> String {
    let duration = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let total_secs = duration.as_secs();
    let millis = duration.subsec_millis();
    let days = (total_secs / 86_400) as i64;
    let day_secs = total_secs % 86_400;
    let hour = day_secs / 3_600;
    let minute = (day_secs % 3_600) / 60;
    let second = day_secs % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02},{millis:03}")
}

// Howard Hinnant's proleptic Gregorian calendar conversion from Unix epoch days.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32, d as u32)
}

macro_rules! log_debug {
    ($id:expr, $($arg:tt)*) => {
        $crate::log::emit($crate::log::Level::Debug, $id, format_args!($($arg)*))
    };
}

macro_rules! log_info {
    ($id:expr, $($arg:tt)*) => {
        $crate::log::emit($crate::log::Level::Info, $id, format_args!($($arg)*))
    };
}

macro_rules! log_warn {
    ($id:expr, $($arg:tt)*) => {
        $crate::log::emit($crate::log::Level::Warning, $id, format_args!($($arg)*))
    };
}

macro_rules! log_error {
    ($id:expr, $($arg:tt)*) => {
        $crate::log::emit($crate::log::Level::Error, $id, format_args!($($arg)*))
    };
}

pub(crate) use {log_debug, log_error, log_info, log_warn};
