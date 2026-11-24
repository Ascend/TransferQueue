// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

//! Native metadata request engine for TransferQueue.

mod engine;
mod log;
mod state;
mod transport;
mod wire;

pub use engine::ControllerEngine;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
