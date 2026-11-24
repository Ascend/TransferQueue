// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::io;
use std::net::IpAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::thread::{self, JoinHandle};

use rmpv::Value;

use crate::Result;
use crate::state::ControllerState;
use crate::transport::Sockets;

pub struct ControllerEngine {
    ports: (u16, u16),
    state: Arc<Mutex<ControllerState>>,
    stopping: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<()>>>,
}

impl ControllerEngine {
    pub fn start(
        ip: IpAddr,
        controller_id: String,
        units: usize,
        preallocate: usize,
    ) -> Result<Self> {
        if units == 0 {
            return Err(io::Error::other("Controller requires at least one storage unit").into());
        }
        let state = Arc::new(Mutex::new(ControllerState::new(units, preallocate)));
        let worker_state = Arc::clone(&state);
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = Arc::clone(&stopping);
        let (ready, ports) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("tq-controller".into())
            .spawn(move || {
                let sockets = Sockets::bind(ip)?;
                ready.send(sockets.ports())?;
                sockets.run(&worker_stopping, &controller_id, &worker_state)
            })?;
        let Ok(ports) = ports.recv() else {
            join(worker)?;
            return Err(io::Error::other("Controller exited before reporting its ports").into());
        };
        Ok(Self {
            ports,
            state,
            stopping,
            worker: Some(worker),
        })
    }

    /// Handshake port followed by the request port.
    pub fn ports(&self) -> (u16, u16) {
        self.ports
    }

    pub fn set_num_storage_units(&self, units: usize) -> Result<()> {
        self.lock_state()
            .set_num_storage_units(units)
            .map_err(io::Error::other)
            .map_err(Into::into)
    }

    pub fn list_partitions(&self) -> Vec<String> {
        self.lock_state().list_partitions()
    }

    pub fn partition_snapshot(&self, name: &str) -> Result<Option<Vec<u8>>> {
        self.lock_state()
            .partition_snapshot(name)
            .map_err(io::Error::other)
            .map_err(Into::into)
    }

    pub fn partition_index_range(&self, name: &str) -> Vec<u64> {
        self.lock_state().partition_index_range(name)
    }

    pub fn mark_clearing(&self, indexes: &[u64], partitions: &[String]) -> Result<()> {
        self.lock_state()
            .mark_clearing_rows(indexes, partitions)
            .map_err(io::Error::other)
            .map_err(Into::into)
    }

    pub fn clear_meta(&self, indexes: &[u64], partitions: &[String]) -> Result<()> {
        self.lock_state()
            .clear_meta_rows(indexes, partitions)
            .map_err(io::Error::other)
            .map_err(Into::into)
    }

    pub fn clear_partition(&self, name: &str) {
        self.lock_state().clear_partition(name);
    }

    pub fn mark_unlinked_public_clearing(
        &self,
        stale_keys: Option<&[String]>,
    ) -> Result<(Vec<u8>, Vec<String>)> {
        let stale: Option<Vec<&str>> =
            stale_keys.map(|keys| keys.iter().map(String::as_str).collect());
        let (metadata, keys) = self.lock_state().mark_unlinked_public(stale.as_deref());
        let mut frame = Vec::new();
        rmpv::encode::write_value(&mut frame, &metadata.encode()?)?;
        Ok((frame, keys))
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.lock_state()
            .save_checkpoint(path)
            .map_err(io::Error::other)
            .map_err(Into::into)
    }

    pub fn validate_checkpoint(path: &str, units: Option<usize>) -> Result<()> {
        ControllerState::read_checkpoint(path, units)
            .map(|_| ())
            .map_err(io::Error::other)
            .map_err(Into::into)
    }

    pub fn load_checkpoint(&self, path: &str) -> Result<()> {
        let units = self.lock_state().num_storage_units();
        let restored =
            ControllerState::read_checkpoint(path, Some(units)).map_err(io::Error::other)?;
        let mut state = self.lock_state();
        if state.num_storage_units() != units {
            return Err(io::Error::other(
                "Rust Controller checkpoint has a different number of storage units",
            )
            .into());
        }
        state.install_checkpoint(restored);
        Ok(())
    }

    pub fn enable_metrics(&self, buckets: Vec<f64>) {
        self.lock_state().enable_metrics(buckets);
    }

    pub fn metrics_snapshot(&self) -> Result<Vec<u8>> {
        let snapshot = Value::Map(self.lock_state().metrics_snapshot());
        let mut frame = Vec::new();
        rmpv::encode::write_value(&mut frame, &snapshot)?;
        Ok(frame)
    }

    pub fn close(&mut self) -> Result<()> {
        if let Some(worker) = self.worker.take() {
            self.stopping.store(true, Ordering::Relaxed);
            join(worker)?;
        }
        Ok(())
    }

    fn lock_state(&self) -> MutexGuard<'_, ControllerState> {
        self.state.lock().expect("controller state poisoned")
    }
}

impl Drop for ControllerEngine {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn join(worker: JoinHandle<Result<()>>) -> Result<()> {
    worker
        .join()
        .map_err(|_| io::Error::other("Controller worker panicked"))?
}
