// Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::sync::Mutex;

use rmpv::Value;

use super::{Bitmap, ControllerState, PUBLIC_PARTITION, StateResult, UNASSIGNED_SEQ, parse_dtype};
use crate::engine::lock_state;

const HEADER: &[u8] = b"TQ-RUST\x01";

impl ControllerState {
    pub(crate) fn encode_checkpoint(&self) -> StateResult<Vec<u8>> {
        let mut bytes = Vec::with_capacity(256);
        bytes.extend_from_slice(HEADER);
        rmp_serde::encode::write_named(&mut bytes, self).map_err(|error| error.to_string())?;
        Ok(bytes)
    }

    pub(crate) fn write_checkpoint_file(path: &str, bytes: &[u8]) -> StateResult<()> {
        let mut file = BufWriter::new(File::create(path).map_err(|error| error.to_string())?);
        file.write_all(bytes).map_err(|error| error.to_string())?;
        file.flush().map_err(|error| error.to_string())
    }

    pub(crate) fn save_checkpoint(&self, path: &str) -> StateResult<()> {
        let bytes = self.encode_checkpoint()?;
        Self::write_checkpoint_file(path, &bytes)
    }

    pub(crate) fn save_checkpoint_unlocked(state: &Mutex<Self>, path: &str) -> StateResult<()> {
        let bytes = lock_state(state)?.encode_checkpoint()?;
        Self::write_checkpoint_file(path, &bytes)
    }

    pub(crate) fn load_checkpoint_unlocked(state: &Mutex<Self>, path: &str) -> StateResult<()> {
        let units = lock_state(state)?.num_storage_units();
        let restored = Self::read_checkpoint(path, Some(units))?;
        let mut guard = lock_state(state)?;
        if guard.num_storage_units() != units {
            return Err(
                "Rust Controller checkpoint has a different number of storage units".into(),
            );
        }
        guard.install_checkpoint(restored);
        Ok(())
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
        restored.validate_invariants(units)?;
        Ok(restored)
    }

    fn validate_invariants(&self, units: Option<usize>) -> StateResult<()> {
        let num_units = self.allocator.reusable.len();
        if num_units == 0 {
            return Err("Rust Controller checkpoint has no storage units".into());
        }
        // `SimpleStorage` maps `global_index % units` to storage units, so changing `units`
        // across checkpoint restore would route existing indexes to the wrong storage units.
        if units.is_some_and(|expected| num_units != expected) {
            return Err(
                "Rust Controller checkpoint has a different number of storage units".into(),
            );
        }

        let mut seen_indexes = HashSet::new();
        for (unit, pool) in self.allocator.reusable.iter().enumerate() {
            for &index in pool {
                if index >= self.allocator.next {
                    return Err("reusable global index exceeds allocator counter".into());
                }
                if (index % num_units as u64) as usize != unit {
                    return Err("reusable global index assigned to wrong storage unit".into());
                }
                if !seen_indexes.insert(index) {
                    return Err("duplicate global index in allocator reusable pool".into());
                }
            }
        }

        let public = self.partitions.get(PUBLIC_PARTITION);
        for (part_name, partition) in &self.partitions {
            let row_len = partition.indexes.len();
            if partition.rows.len() != row_len || partition.seqs.len() != row_len {
                return Err(format!(
                    "partition {part_name:?} has inconsistent row/index/seq lengths"
                ));
            }
            validate_bitmap(&partition.active, row_len, "active")?;
            for consumed in partition.consumed.values() {
                validate_bitmap(consumed, row_len, "consumed")?;
            }

            let mut seen_seqs = HashSet::new();
            let mut inactive_count = 0usize;
            for (row, &index) in partition.indexes.iter().enumerate() {
                if partition.rows.get(&index) != Some(&row) {
                    return Err(format!(
                        "partition {part_name:?} has corrupt row mapping for index {index}"
                    ));
                }
                if index >= self.allocator.next {
                    return Err(format!(
                        "partition {part_name:?} index {index} exceeds allocator counter"
                    ));
                }
                if !seen_indexes.insert(index) {
                    return Err(format!(
                        "duplicate global index {index} across partitions or reusable pool"
                    ));
                }
                let seq = partition.seqs[row];
                if partition.active.get(row) {
                    if seq == UNASSIGNED_SEQ || seq >= partition.next_seq || !seen_seqs.insert(seq)
                    {
                        return Err(format!(
                            "partition {part_name:?} has invalid sequence ordinal at row {row}"
                        ));
                    }
                } else {
                    inactive_count += 1;
                    if !partition.reserved.contains(&index) {
                        return Err(format!(
                            "partition {part_name:?} has unreserved inactive index {index}"
                        ));
                    }
                }
            }
            if partition.reserved.len() != inactive_count {
                return Err(format!(
                    "partition {part_name:?} reserved set disagrees with inactive rows"
                ));
            }

            if partition.keys.len() != partition.index_keys.len() {
                return Err(format!(
                    "partition {part_name:?} has mismatched key mappings"
                ));
            }
            for (key, &index) in &partition.keys {
                let Some(&row) = partition.rows.get(&index) else {
                    return Err(format!(
                        "partition {part_name:?} key {key:?} references unknown index {index}"
                    ));
                };
                if !partition.active.get(row) || partition.index_keys.get(&index) != Some(key) {
                    return Err(format!(
                        "partition {part_name:?} key {key:?} has invalid reverse or active state"
                    ));
                }
            }

            for (field_name, field_state) in &partition.fields {
                validate_bitmap(&field_state.present, row_len, "present")?;
                validate_bitmap(&field_state.ready, row_len, "ready")?;
                for word in 0..row_len.div_ceil(64) {
                    let active_bits = partition.active.word(word);
                    let present_bits = field_state.present.word(word);
                    let ready_bits = field_state.ready.word(word);
                    if (ready_bits & !present_bits) != 0 || (present_bits & !active_bits) != 0 {
                        return Err(format!(
                            "partition {part_name:?} field {field_name:?} has ready/present bits on inactive rows"
                        ));
                    }
                }
                let any_present = field_state.present.0.iter().any(|&word| word != 0);
                if any_present && field_state.schema.is_none() {
                    return Err(format!(
                        "partition {part_name:?} field {field_name:?} has present rows without schema"
                    ));
                }
                if let Some(schema) = &field_state.schema {
                    if let Some(dtype) = &schema.dtype {
                        parse_dtype(&Value::from(dtype.as_str()), field_name)?;
                    }
                    for index in schema.sample_shapes.keys() {
                        if !partition.rows.contains_key(index) {
                            return Err(format!(
                                "partition {part_name:?} field {field_name:?} has sample shape for unknown index {index}"
                            ));
                        }
                    }
                }
            }

            for (&index, raw) in &partition.custom_meta {
                if !partition.rows.contains_key(&index) {
                    return Err(format!(
                        "partition {part_name:?} has custom_meta for unknown index {index}"
                    ));
                }
                let decoded = raw.decode()?;
                let Value::Map(entries) = decoded else {
                    return Err(format!(
                        "partition {part_name:?} custom_meta for index {index} must be a map"
                    ));
                };
                if entries.iter().any(|(k, _)| !k.is_str()) {
                    return Err(format!(
                        "partition {part_name:?} custom_meta for index {index} has non-string key"
                    ));
                }
            }

            for (&index, fields) in &partition.backend_meta {
                if !partition.rows.contains_key(&index) {
                    return Err(format!(
                        "partition {part_name:?} has backend_meta for unknown index {index}"
                    ));
                }
                for raw in fields.values() {
                    raw.decode()?;
                }
            }

            if !partition.links.is_empty() {
                if part_name == PUBLIC_PARTITION {
                    return Err("public partition cannot hold outgoing links".into());
                }
                let public = public.ok_or("checkpoint has links without public partition")?;
                for (&index, row_links) in &partition.links {
                    let Some(&row) = partition.rows.get(&index) else {
                        return Err(format!(
                            "partition {part_name:?} has links for unknown index {index}"
                        ));
                    };
                    if !partition.active.get(row) {
                        return Err(format!(
                            "partition {part_name:?} has links on inactive index {index}"
                        ));
                    }
                    for (field_name, &source) in row_links {
                        let Some(&pub_row) = public.rows.get(&source) else {
                            return Err(format!(
                                "partition {part_name:?} links field {field_name:?} to missing public index {source}"
                            ));
                        };
                        if !public.active.get(pub_row) {
                            return Err(format!(
                                "partition {part_name:?} links field {field_name:?} to inactive public index {source}"
                            ));
                        }
                        if partition
                            .fields
                            .get(field_name)
                            .and_then(|f| f.schema.as_ref())
                            .is_none()
                            || public
                                .fields
                                .get(field_name)
                                .and_then(|f| f.schema.as_ref())
                                .is_none()
                        {
                            return Err(format!(
                                "linked field {field_name:?} is missing schema in checkpoint"
                            ));
                        }
                    }
                }
            }

            if part_name == PUBLIC_PARTITION {
                for &index in &partition.ever_linked {
                    let Some(&row) = partition.rows.get(&index) else {
                        return Err(format!(
                            "public partition ever_linked references unknown index {index}"
                        ));
                    };
                    if !partition.active.get(row) {
                        return Err(format!(
                            "public partition ever_linked references inactive index {index}"
                        ));
                    }
                }
            } else if !partition.ever_linked.is_empty() {
                return Err(format!(
                    "non-public partition {part_name:?} must not populate ever_linked"
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn install_checkpoint(&mut self, mut restored: Self) {
        let units = restored.allocator.reusable.len();
        for partition in restored.partitions.values_mut() {
            partition.loads = vec![0; units];
            for &index in &partition.indexes {
                partition.loads[(index % units as u64) as usize] += 1;
            }
            partition.linked_by.clear();
            for (&index, links) in &partition.links {
                for &source in links.values() {
                    partition.linked_by.entry(source).or_default().insert(index);
                }
            }
            let mut prev_seq = None;
            partition.seq_sorted = (0..partition.indexes.len()).all(|row| {
                if !partition.active.get(row) {
                    return true;
                }
                let seq = partition.seqs[row];
                let ordered = prev_seq.is_none_or(|prev| prev < seq);
                prev_seq = Some(seq);
                ordered
            });
            partition.consumed_cursor.clear();
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

fn validate_bitmap(bitmap: &Bitmap, row_len: usize, label: &str) -> StateResult<()> {
    let max_words = row_len.div_ceil(64);
    if bitmap.0.len() > max_words {
        return Err(format!(
            "{label} bitmap word length {} exceeds row capacity {row_len}",
            bitmap.0.len()
        ));
    }
    let tail_bits = row_len % 64;
    if tail_bits != 0 && bitmap.0.len() == max_words && (bitmap.0[max_words - 1] >> tail_bits) != 0
    {
        return Err(format!(
            "{label} bitmap has set bits beyond row count {row_len}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::Request;
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
    fn checkpoint_roundtrip_preserves_observable_state_loads_and_allocator_continuity() {
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

        // Runtime settings (`preallocate` and `metrics`), derived `loads`, and reverse link index survive restore.
        assert_eq!(target.preallocate, 9);
        assert!(target.metrics.is_some());
        assert_eq!(
            target.partitions["train"].loads.iter().sum::<usize>(),
            target.partitions["train"].indexes.len()
        );
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
    fn checkpoint_rejects_missing_corrupt_or_broken_invariants() {
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

        // 1. Out-of-bounds `active` bitmap bit is rejected.
        let mut bad_bitmap = ControllerState::read_checkpoint(ckpt.path(), Some(2)).unwrap();
        bad_bitmap.partitions.get_mut("p0").unwrap().active.0 = vec![u64::MAX];
        let mut raw = HEADER.to_vec();
        rmp_serde::encode::write_named(&mut raw, &bad_bitmap).unwrap();
        std::fs::write(ckpt.path(), &raw).unwrap();
        assert!(
            ControllerState::read_checkpoint(ckpt.path(), Some(2))
                .err()
                .unwrap()
                .contains("bitmap")
        );

        // 2. Overlapping reusable and active `GlobalIndex` is rejected.
        let mut overlap = ControllerState::new(2, 1);
        let meta = overlap.kv_retrieve_meta("p0", &["k0"], true).unwrap();
        overlap.allocator.reusable[(meta.indexes[0] % 2) as usize].push_back(meta.indexes[0]);
        let mut raw = HEADER.to_vec();
        rmp_serde::encode::write_named(&mut raw, &overlap).unwrap();
        std::fs::write(ckpt.path(), &raw).unwrap();
        assert!(
            ControllerState::read_checkpoint(ckpt.path(), Some(2))
                .err()
                .unwrap()
                .contains("duplicate global index")
        );

        // 3. Corrupt `rows` mapping is rejected.
        let mut bad_rows = ControllerState::new(2, 1);
        bad_rows.kv_retrieve_meta("p0", &["k0"], true).unwrap();
        bad_rows.partitions.get_mut("p0").unwrap().rows.clear();
        let mut raw = HEADER.to_vec();
        rmp_serde::encode::write_named(&mut raw, &bad_rows).unwrap();
        std::fs::write(ckpt.path(), &raw).unwrap();
        assert!(
            ControllerState::read_checkpoint(ckpt.path(), Some(2))
                .err()
                .unwrap()
                .contains("inconsistent row/index/seq")
        );

        valid.save_checkpoint(ckpt.path()).unwrap();
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
