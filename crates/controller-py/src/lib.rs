// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Display;
use std::net::IpAddr;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tq_controller::ControllerEngine as Engine;

#[pyclass(module = "transfer_queue._controller_engine")]
struct ControllerEngine {
    engine: Engine,
}

// Release the GIL via `py.detach` around worker startup/shutdown, snapshot serialization,
// and checkpoint file I/O; fast in-memory metadata calls stay synchronous on the actor thread.
#[pymethods]
impl ControllerEngine {
    #[new]
    #[pyo3(signature = (ip, controller_id, storage_units=1, preallocate=1))]
    fn new(
        py: Python<'_>,
        ip: &str,
        controller_id: String,
        storage_units: usize,
        preallocate: usize,
    ) -> PyResult<Self> {
        let ip = ip.parse::<IpAddr>().map_err(|error| {
            PyValueError::new_err(format!("Invalid Controller bind address {ip:?}: {error}"))
        })?;
        let engine = py
            .detach(move || Engine::start(ip, controller_id, storage_units, preallocate))
            .map_err(|error| {
                PyRuntimeError::new_err(format!("Rust Controller failed to start: {error}"))
            })?;
        Ok(Self { engine })
    }

    fn ports(&self) -> (u16, u16) {
        self.engine.ports()
    }

    fn set_num_storage_units(&self, units: usize) -> PyResult<()> {
        self.engine
            .set_num_storage_units(units)
            .map_err(runtime_err)
    }

    fn list_partitions(&self) -> Vec<String> {
        self.engine.list_partitions()
    }

    fn partition_snapshot<'py>(
        &self,
        py: Python<'py>,
        partition_id: &str,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let snapshot = py
            .detach(|| self.engine.partition_snapshot(partition_id))
            .map_err(runtime_err)?;
        Ok(snapshot.map(|bytes| PyBytes::new(py, &bytes)))
    }

    fn partition_index_range(&self, partition_id: &str) -> Vec<u64> {
        self.engine.partition_index_range(partition_id)
    }

    fn mark_clearing(&self, global_indexes: Vec<u64>, partition_ids: Vec<String>) -> PyResult<()> {
        self.engine
            .mark_clearing(&global_indexes, &partition_ids)
            .map_err(runtime_err)
    }

    fn clear_meta(&self, global_indexes: Vec<u64>, partition_ids: Vec<String>) -> PyResult<()> {
        self.engine
            .clear_meta(&global_indexes, &partition_ids)
            .map_err(runtime_err)
    }

    fn clear_partition(&self, partition_id: &str) {
        self.engine.clear_partition(partition_id);
    }

    #[pyo3(signature = (stale_keys=None))]
    fn mark_unlinked_public_clearing<'py>(
        &self,
        py: Python<'py>,
        stale_keys: Option<Vec<String>>,
    ) -> PyResult<(Bound<'py, PyBytes>, Vec<String>)> {
        let (metadata, keys) = py
            .detach(|| {
                self.engine
                    .mark_unlinked_public_clearing(stale_keys.as_deref())
            })
            .map_err(runtime_err)?;
        Ok((PyBytes::new(py, &metadata), keys))
    }

    fn save_checkpoint(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.engine.save_checkpoint(path))
            .map_err(runtime_err)
    }

    fn load_checkpoint(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.engine.load_checkpoint(path))
            .map_err(runtime_err)
    }

    fn enable_metrics(&self, buckets: Vec<f64>) {
        self.engine.enable_metrics(buckets);
    }

    fn metrics_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let snapshot = py
            .detach(|| self.engine.metrics_snapshot())
            .map_err(runtime_err)?;
        Ok(PyBytes::new(py, &snapshot))
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.engine.close()).map_err(|error| {
            PyRuntimeError::new_err(format!("Rust Controller failed to close: {error}"))
        })
    }
}

#[pyfunction]
#[pyo3(signature = (path, storage_units=None))]
fn validate_checkpoint(py: Python<'_>, path: &str, storage_units: Option<usize>) -> PyResult<()> {
    py.detach(|| Engine::validate_checkpoint(path, storage_units))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn runtime_err(error: impl Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pymodule]
fn _controller_engine(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ControllerEngine>()?;
    module.add_function(wrap_pyfunction!(validate_checkpoint, module)?)
}
