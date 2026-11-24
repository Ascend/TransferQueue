// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

use super::{ControllerState, StateResult};

const HEADER: &[u8] = b"TQ-RUST\x01";

impl ControllerState {
    pub(crate) fn save_checkpoint(&self, path: &str) -> StateResult<()> {
        let mut file = BufWriter::new(File::create(path).map_err(|error| error.to_string())?);
        file.write_all(HEADER).map_err(|error| error.to_string())?;
        rmp_serde::encode::write_named(&mut file, self).map_err(|error| error.to_string())?;
        file.flush().map_err(|error| error.to_string())
    }

    pub(crate) fn read_checkpoint(path: &str, units: Option<usize>) -> StateResult<Self> {
        let mut file = BufReader::new(File::open(path).map_err(|error| error.to_string())?);
        let mut header = [0; HEADER.len()];
        if file.read_exact(&mut header).is_err() || header.as_slice() != HEADER {
            return Err("unsupported Rust Controller checkpoint format".into());
        }
        let restored: Self = rmp_serde::from_read(&mut file).map_err(|error| error.to_string())?;
        if !file
            .fill_buf()
            .map_err(|error| error.to_string())?
            .is_empty()
        {
            return Err("trailing bytes in Rust Controller checkpoint".into());
        }
        if restored.allocator.reusable.is_empty() {
            return Err("Rust Controller checkpoint has no storage units".into());
        }
        // `SimpleStorage` maps `global_index % units` to storage units, so changing `units`
        // across checkpoint restore would route existing indexes to the wrong storage units.
        if units.is_some_and(|units| restored.allocator.reusable.len() != units) {
            return Err(
                "Rust Controller checkpoint has a different number of storage units".into(),
            );
        }
        Ok(restored)
    }

    pub(crate) fn install_checkpoint(&mut self, mut restored: Self) {
        for partition in restored.partitions.values_mut() {
            partition.linked_by.clear();
            for (&index, links) in &partition.links {
                for &source in links.values() {
                    partition.linked_by.entry(source).or_default().insert(index);
                }
            }
        }
        restored.preallocate = self.preallocate;
        restored.metrics = self.metrics.take();
        *self = restored;
    }

    pub(super) fn load_checkpoint(&mut self, path: &str) -> StateResult<()> {
        let restored = Self::read_checkpoint(path, Some(self.allocator.reusable.len()))?;
        self.install_checkpoint(restored);
        Ok(())
    }

    pub(crate) fn partition_snapshot(&self, name: &str) -> StateResult<Option<Vec<u8>>> {
        self.partitions
            .get(name)
            .map(|partition| rmp_serde::to_vec_named(partition).map_err(|error| error.to_string()))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::Request;
    use rmpv::Value;
    use std::path::PathBuf;
    use uuid::Uuid;

    struct TempFile(PathBuf);

    impl TempFile {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!("tq-rust-ckpt-{}.msgpack", Uuid::new_v4())))
        }

        fn path(&self) -> &str {
            self.0.to_str().unwrap()
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn checkpoint_roundtrip_preserves_observable_state_and_allocator_continuity() {
        let ckpt = TempFile::new();
        let pub_part = super::super::links::PUBLIC_PARTITION;
        let mut source = ControllerState::new(2, 3);
        let pub_meta = source.kv_retrieve_meta(pub_part, &["pk"], true).unwrap();
        source
            .execute(&Request {
                request_type: "NOTIFY_DATA_UPDATE".into(),
                sender_id: "s".into(),
                body: vec![
                    ("partition_id".into(), pub_part.into()),
                    (
                        "global_indexes".into(),
                        Value::Array(vec![pub_meta.indexes[0].into()]),
                    ),
                    (
                        "field_schema".into(),
                        Value::Map(vec![(
                            "prompt".into(),
                            Value::Map(vec![
                                ("dtype".into(), "int64".into()),
                                ("shape".into(), Value::Array(vec![2.into()])),
                            ]),
                        )]),
                    ),
                ],
            })
            .unwrap();
        source
            .kv_link(&vec![
                ("partition_id".into(), "train".into()),
                ("keys".into(), Value::Array(vec!["k0".into(), "k1".into()])),
                (
                    "public_keys".into(),
                    Value::Array(vec!["pk".into(), "pk".into()]),
                ),
                ("fields".into(), Value::Array(vec!["prompt".into()])),
            ])
            .unwrap();
        let linked = source
            .kv_retrieve_meta("train", &["k0", "k1"], false)
            .unwrap();
        source
            .clear_meta_rows(&[linked.indexes[1]], &["train".to_string()])
            .unwrap();
        source.save_checkpoint(ckpt.path()).unwrap();

        let mut target = ControllerState::new(2, 9);
        target.enable_metrics(vec![0.1]);
        target.load_checkpoint(ckpt.path()).unwrap();

        // Runtime settings (`preallocate` and `metrics`) and reverse link index survive restore.
        assert_eq!(target.preallocate, 9);
        assert!(target.metrics.is_some());
        assert!(target.kv_retrieve_meta(pub_part, &["pk"], true).is_err());
        assert_eq!(
            target
                .kv_retrieve_meta("train", &["k0"], false)
                .unwrap()
                .indexes,
            vec![linked.indexes[0]]
        );

        // New allocations never collide with restored active indexes (`pub_meta` or `k0`).
        let fresh = target
            .kv_retrieve_meta("eval", &["e0", "e1"], true)
            .unwrap();
        assert!(!fresh.indexes.contains(&pub_meta.indexes[0]));
        assert!(!fresh.indexes.contains(&linked.indexes[0]));

        // `ever_linked` survives restore so clearing `train` reclaims orphaned `pk`.
        target.clear_partition("train");
        assert_eq!(target.mark_unlinked_public(None).1, vec!["pk".to_string()]);
    }

    #[test]
    fn checkpoint_rejects_missing_corrupt_or_mismatched_files() {
        let ckpt = TempFile::new();
        assert!(ControllerState::read_checkpoint(ckpt.path(), Some(2)).is_err());

        let mut valid = ControllerState::new(2, 1);
        valid.kv_retrieve_meta("p0", &["k0"], true).unwrap();
        valid.save_checkpoint(ckpt.path()).unwrap();
        assert!(ControllerState::read_checkpoint(ckpt.path(), None).is_ok());

        assert!(
            ControllerState::read_checkpoint(ckpt.path(), Some(3))
                .err()
                .unwrap()
                .contains("different number of storage units")
        );

        let bytes = std::fs::read(ckpt.path()).unwrap();
        let mut trailing = bytes.clone();
        trailing.push(0x00);
        std::fs::write(ckpt.path(), &trailing).unwrap();
        assert!(
            ControllerState::read_checkpoint(ckpt.path(), Some(2))
                .err()
                .unwrap()
                .contains("trailing bytes")
        );

        std::fs::write(ckpt.path(), &bytes[..HEADER.len() + 2]).unwrap();
        assert!(ControllerState::read_checkpoint(ckpt.path(), Some(2)).is_err());

        std::fs::write(ckpt.path(), b"TQ-RUST\x02invalid").unwrap();
        assert!(
            ControllerState::read_checkpoint(ckpt.path(), Some(2))
                .err()
                .unwrap()
                .contains("unsupported Rust Controller checkpoint format")
        );
    }
}
