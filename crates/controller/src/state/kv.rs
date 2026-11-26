// Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use super::{
    ControllerState, LocalRow, PUBLIC_PARTITION, PartitionState, StateResult, flag, strings,
};
use crate::wire::{BatchMeta, Map, field, text};
use rmpv::Value;

impl PartitionState {
    pub(super) fn rows_for_keys(&self, keys: &[&str]) -> Vec<LocalRow> {
        keys.iter()
            .filter_map(|key| self.keys.get(*key).map(|index| self.rows[index]))
            .collect()
    }

    pub(super) fn fields_ready_in_all(&self, rows: &[LocalRow]) -> Vec<&str> {
        self.fields
            .iter()
            .filter(|(_, field)| rows.iter().all(|&row| field.ready.get(row)))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    pub(super) fn clear_fields(&mut self, rows: &[LocalRow], fields: &[&str]) {
        for &row in rows {
            let index = self.indexes[row];
            self.drop_links(index, Some(fields));
            if let Some(backend) = self.backend_meta.get_mut(&index) {
                for &name in fields {
                    backend.remove(name);
                }
                if backend.is_empty() {
                    self.backend_meta.remove(&index);
                }
            }
        }
        for name in fields {
            let field = &mut self.fields[*name];
            for &row in rows {
                let index = self.indexes[row];
                field.ready.set(row, false);
                field.present.set(row, false);
                if let Some(schema) = &mut field.schema {
                    schema.sample_shapes.remove(&index);
                }
            }
            field.remove_empty_schema();
        }
    }
}

impl ControllerState {
    pub(super) fn register_keys(&mut self, name: &str, keys: &[&str]) {
        self.create_partition(name);
        let partition = &mut self.partitions[name];
        for &key in keys {
            if !partition.keys.contains_key(key) {
                let row = partition.insert(1, &mut self.allocator)[0];
                let index = partition.indexes[row];
                partition.keys.insert(key.to_owned(), index);
                partition.index_keys.insert(index, key.to_owned());
            }
        }
    }

    pub(super) fn kv_retrieve_meta(
        &mut self,
        name: &str,
        keys: &[&str],
        create: bool,
    ) -> StateResult<BatchMeta> {
        if create {
            if name == PUBLIC_PARTITION
                && let Some(public) = self.partitions.get(name)
            {
                let indexes: Vec<_> = keys
                    .iter()
                    .filter_map(|&key| public.keys.get(key).copied())
                    .collect();
                self.check_public_not_linked(&indexes, None)?;
            }
            self.register_keys(name, keys);
        }
        let Some(partition) = self.partitions.get(name) else {
            return Ok(BatchMeta::default());
        };
        let rows = partition.rows_for_keys(keys);
        let fields = partition.fields_ready_in_all(&rows);
        Ok(partition.snapshot(name, &rows, &fields, "force_fetch"))
    }

    pub(super) fn kv_list(&self, body: &Map) -> StateResult<Map> {
        let requested = field(body, "partition_id")
            .filter(|name| !name.is_nil())
            .map(|name| name.as_str().ok_or("partition_id must be a string"))
            .transpose()?;
        let partition_info = self
            .partitions
            .iter()
            .filter(|(name, _)| requested.is_none_or(|requested| *name == requested))
            .map(|(name, partition)| {
                let keys = partition
                    .keys
                    .iter()
                    .map(|(key, index)| {
                        let tag = match partition.custom_meta.get(index) {
                            Some(raw) => raw.decode()?,
                            None => Value::Map(Vec::new()),
                        };
                        Ok((key.as_str().into(), tag))
                    })
                    .collect::<StateResult<Vec<_>>>()?;
                Ok((name.as_str().into(), Value::Map(keys)))
            })
            .collect::<StateResult<Vec<_>>>()?;
        let message = match requested {
            Some(name) if !self.partitions.contains_key(name) => {
                format!("partition {name} does not exist")
            }
            _ => "success".into(),
        };
        Ok(vec![
            ("partition_info".into(), Value::Map(partition_info)),
            ("message".into(), message.into()),
        ])
    }

    pub(super) fn kv_exists(&self, body: &Map) -> StateResult<Map> {
        let name = text(body, "partition_id")?;
        let keys = strings(body, "keys")?;
        let mut ready = Vec::new();
        if let Some(partition) = self.partitions.get(name) {
            for key in keys {
                let Some(index) = partition.keys.get(key) else {
                    continue;
                };
                let row = partition.rows[index];
                let mut fields: Vec<_> = partition
                    .fields
                    .iter()
                    .filter(|(_, field)| field.ready.get(row))
                    .map(|(name, _)| name.as_str())
                    .collect();
                fields.sort_unstable();
                ready.push((
                    key.into(),
                    Value::Array(fields.into_iter().map(Value::from).collect()),
                ));
            }
        }
        Ok(vec![("ready".into(), Value::Map(ready))])
    }

    pub(super) fn kv_describe(&self, body: &Map) -> StateResult<Map> {
        let name = text(body, "partition_id")?;
        let keys = strings(body, "keys")?;
        let partition = self
            .partitions
            .get(name)
            .ok_or_else(|| format!("partition {name:?} does not exist"))?;
        let missing: Vec<_> = keys
            .iter()
            .filter(|&&key| !partition.keys.contains_key(key))
            .collect();
        if !missing.is_empty() {
            return Err(format!("keys not found in partition {name:?}: {missing:?}"));
        }

        let mut rows = Vec::with_capacity(keys.len());
        let mut schema_names = BTreeSet::new();
        for key in keys {
            let index = partition.keys[key];
            let row = partition.rows[&index];
            let mut present: Vec<_> = partition
                .fields
                .iter()
                .filter(|(_, field)| field.present.get(row))
                .map(|(name, field)| (name.as_str(), field.ready.get(row)))
                .collect();
            present.sort_unstable_by_key(|&(name, _)| name);
            let links = partition.links.get(&index);
            schema_names.extend(present.iter().filter_map(|&(name, _)| {
                (!links.is_some_and(|links| links.contains_key(name))).then_some(name)
            }));
            let links = match links {
                Some(links) => {
                    let public = self
                        .partitions
                        .get(PUBLIC_PARTITION)
                        .ok_or("public partition does not exist")?;
                    links
                        .iter()
                        .map(|(name, source)| {
                            let target = public
                                .index_keys
                                .get(source)
                                .ok_or_else(|| format!("public row {source} has no key"))?;
                            Ok((name.as_str().into(), target.as_str().into()))
                        })
                        .collect::<StateResult<Vec<_>>>()?
                }
                None => Vec::new(),
            };
            let tag = match partition.custom_meta.get(&index) {
                Some(raw) => raw.decode()?,
                None => Value::Map(Vec::new()),
            };
            let unfinalized = present
                .iter()
                .filter(|(_, ready)| !ready)
                .map(|&(name, _)| name.into())
                .collect();
            rows.push((
                key.into(),
                Value::Map(vec![
                    ("global_index".into(), index.into()),
                    (
                        "fields".into(),
                        Value::Array(present.into_iter().map(|(name, _)| name.into()).collect()),
                    ),
                    ("unfinalized".into(), Value::Array(unfinalized)),
                    ("links".into(), Value::Map(links)),
                    ("tag".into(), tag),
                ]),
            ));
        }
        let schema = schema_names
            .into_iter()
            .filter_map(|name| {
                let schema = partition.fields[name].schema.as_ref()?;
                Some((
                    name.into(),
                    Value::Map(vec![
                        ("is_nested".into(), schema.nested.into()),
                        ("is_non_tensor".into(), schema.non_tensor.into()),
                    ]),
                ))
            })
            .collect();
        Ok(vec![
            ("rows".into(), Value::Map(rows)),
            ("field_schema".into(), Value::Map(schema)),
        ])
    }

    // Reject linked fields before allocating keys so shared `PUBLIC_PARTITION` payloads cannot be
    // mutated through a consumer partition, and report `held` per row so StorageManager knows
    // whether to merge into an existing value or write a new one.
    pub(super) fn kv_prepare_update(&mut self, body: &Map) -> StateResult<(BatchMeta, Map)> {
        let name = text(body, "partition_id")?;
        let keys = strings(body, "keys")?;
        let fields = strings(body, "fields")?;
        if let Some(partition) = self.partitions.get(name) {
            let linked: Vec<_> = keys
                .iter()
                .filter(|&&key| {
                    partition
                        .keys
                        .get(key)
                        .and_then(|index| partition.links.get(index))
                        .is_some_and(|links| fields.iter().any(|&name| links.contains_key(name)))
                })
                .collect();
            if !linked.is_empty() {
                return Err(format!(
                    "kv_update cannot merge into fields linked by kv_link on keys {linked:?}"
                ));
            }
        }
        let metadata = self.kv_retrieve_meta(name, &keys, true)?;
        let partition = &self.partitions[name];
        let merge_plan = fields
            .into_iter()
            .map(|name| {
                let field = partition.fields.get(name);
                let schema = field.and_then(|field| field.schema.as_ref());
                let held = metadata
                    .indexes
                    .iter()
                    .map(|index| {
                        field
                            .is_some_and(|field| field.present.get(partition.rows[index]))
                            .into()
                    })
                    .collect();
                (
                    name.into(),
                    Value::Map(vec![
                        (
                            "dtype".into(),
                            schema
                                .and_then(|schema| schema.dtype.as_deref())
                                .map_or(Value::Nil, Value::from),
                        ),
                        (
                            "is_non_tensor".into(),
                            schema.map_or(Value::Nil, |schema| schema.non_tensor.into()),
                        ),
                        ("held".into(), Value::Array(held)),
                    ]),
                )
            })
            .collect();
        Ok((metadata, merge_plan))
    }

    pub(super) fn kv_clear_fields(&mut self, body: &Map) -> StateResult<BatchMeta> {
        let name = text(body, "partition_id")?;
        let keys = strings(body, "keys")?;
        let mut fields = strings(body, "fields")?;
        let Some(partition) = self.partitions.get(name) else {
            return Ok(BatchMeta::default());
        };
        let rows = partition.rows_for_keys(&keys);
        fields.retain(|name| {
            partition
                .fields
                .get(*name)
                .is_some_and(|field| field.schema.is_some())
        });
        if rows.is_empty() || fields.is_empty() {
            return Ok(BatchMeta::default());
        }
        if name == PUBLIC_PARTITION {
            let indexes: Vec<_> = rows.iter().map(|&row| partition.indexes[row]).collect();
            self.check_public_not_linked(&indexes, Some(&fields))?;
        }
        let metadata = partition.snapshot(name, &rows, &fields, "force_fetch");
        self.partitions[name].clear_fields(&rows, &fields);
        Ok(metadata)
    }

    pub(super) fn validate_dump_schema(&self, body: &Map) -> StateResult<()> {
        let name = text(body, "partition_id")?;
        let Some(partition) = self.partitions.get(name) else {
            return Ok(());
        };
        let fields = field(body, "field_schema")
            .and_then(Value::as_map)
            .ok_or("missing field_schema")?;
        for (name, incoming) in fields {
            let name = name.as_str().ok_or("field name must be a string")?;
            let Some(schema) = partition
                .fields
                .get(name)
                .and_then(|field| field.schema.as_ref())
            else {
                continue;
            };
            let incoming = incoming.as_map().ok_or("field schema must be a map")?;
            if schema.non_tensor != flag(incoming, "is_non_tensor", false)? {
                return Err(format!("Field {name:?} tensor/non-tensor type mismatch"));
            }
            let incoming_dtype = match field(incoming, "dtype") {
                None | Some(Value::Nil) => None,
                Some(value) => Some(super::parse_dtype(value, name)?),
            };
            if let (Some(previous), Some(incoming)) = (schema.dtype.as_deref(), incoming_dtype)
                && previous != incoming
            {
                return Err(format!(
                    "Field '{name}' dtype mismatch: torch.{previous} != torch.{incoming}"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{Body, Request};

    fn strs(items: &[&str]) -> Value {
        Value::Array(items.iter().copied().map(Value::from).collect())
    }

    fn tensor_schema(dtype: &str, shape: &[u64]) -> Value {
        Value::Map(vec![
            ("dtype".into(), dtype.into()),
            (
                "shape".into(),
                Value::Array(shape.iter().copied().map(Value::from).collect()),
            ),
            ("is_nested".into(), false.into()),
            ("is_non_tensor".into(), false.into()),
        ])
    }

    fn notify(state: &mut ControllerState, partition: &str, idx: u64, schema: Map, ready: bool) {
        state
            .execute(&Request {
                request_type: "NOTIFY_DATA_UPDATE".into(),
                sender_id: "test".into(),
                body: vec![
                    ("partition_id".into(), partition.into()),
                    ("global_indexes".into(), Value::Array(vec![idx.into()])),
                    ("field_schema".into(), Value::Map(schema)),
                    ("ready".into(), ready.into()),
                ],
            })
            .unwrap();
    }

    fn ready_fields(state: &ControllerState, partition: &str, key: &str) -> Value {
        let res = state
            .kv_exists(&vec![
                ("partition_id".into(), partition.into()),
                ("keys".into(), strs(&[key])),
            ])
            .unwrap();
        field(field(&res, "ready").unwrap().as_map().unwrap(), key)
            .unwrap()
            .clone()
    }

    #[test]
    fn kv_lifecycle_and_unfinalized_update_visibility() {
        let mut state = ControllerState::new(2, 1);

        // Read-only retrieve on a missing partition neither creates it nor allocates indexes.
        assert!(
            state
                .kv_retrieve_meta("p_kv", &["k0"], false)
                .unwrap()
                .indexes
                .is_empty()
        );
        assert!(state.list_partitions().is_empty());

        let (prepared, plan) = state
            .kv_prepare_update(&vec![
                ("partition_id".into(), "p_kv".into()),
                ("keys".into(), strs(&["k0", "k1"])),
                ("fields".into(), strs(&["ids"])),
            ])
            .unwrap();
        let ids_plan = field(&plan, "ids").unwrap().as_map().unwrap();
        assert_eq!(
            field(ids_plan, "held").unwrap(),
            &Value::Array(vec![false.into(), false.into()])
        );

        let k0_idx = prepared.indexes[0];
        notify(
            &mut state,
            "p_kv",
            k0_idx,
            vec![("ids".into(), tensor_schema("int64", &[4]))],
            false,
        );

        // Unfinalized writes stay invisible to consumers while `kv_describe` and `kv_prepare_update` track them.
        assert_eq!(ready_fields(&state, "p_kv", "k0"), strs(&[]));
        let hidden = state.kv_retrieve_meta("p_kv", &["k0"], false).unwrap();
        assert!(hidden.schema.is_empty() && hidden.status == vec![0]);

        let (_, fetch_body) = state
            .execute(&Request {
                request_type: "GET_META".into(),
                sender_id: "test".into(),
                body: vec![
                    ("partition_id".into(), "p_kv".into()),
                    ("data_fields".into(), strs(&["ids"])),
                    ("mode".into(), "fetch".into()),
                    ("task_name".into(), "consumer".into()),
                    ("batch_size".into(), 1.into()),
                ],
            })
            .unwrap();
        let Body::Metadata(fetched, _) = fetch_body else {
            panic!("expected metadata");
        };
        assert!(fetched.indexes.is_empty());

        let desc = state
            .kv_describe(&vec![
                ("partition_id".into(), "p_kv".into()),
                ("keys".into(), strs(&["k0"])),
            ])
            .unwrap();
        let k0_row = field(field(&desc, "rows").unwrap().as_map().unwrap(), "k0")
            .unwrap()
            .as_map()
            .unwrap();
        assert_eq!(field(k0_row, "fields").unwrap(), &strs(&["ids"]));
        assert_eq!(field(k0_row, "unfinalized").unwrap(), &strs(&["ids"]));

        let (_, next_plan) = state
            .kv_prepare_update(&vec![
                ("partition_id".into(), "p_kv".into()),
                ("keys".into(), strs(&["k0", "k1"])),
                ("fields".into(), strs(&["ids"])),
            ])
            .unwrap();
        let ids_next = field(&next_plan, "ids").unwrap().as_map().unwrap();
        assert_eq!(field(ids_next, "dtype").unwrap().as_str(), Some("int64"));
        assert_eq!(
            field(ids_next, "held").unwrap(),
            &Value::Array(vec![true.into(), false.into()])
        );

        notify(
            &mut state,
            "p_kv",
            k0_idx,
            vec![("ids".into(), tensor_schema("int64", &[4]))],
            true,
        );
        assert_eq!(ready_fields(&state, "p_kv", "k0"), strs(&["ids"]));
    }

    #[test]
    fn kv_clear_fields_and_validate_dump_schema() {
        let mut state = ControllerState::new(1, 1);
        let idx = state
            .kv_retrieve_meta("p_dump", &["k0"], true)
            .unwrap()
            .indexes[0];
        notify(
            &mut state,
            "p_dump",
            idx,
            vec![
                ("ids".into(), tensor_schema("int64", &[4])),
                ("mask".into(), tensor_schema("bool", &[4])),
            ],
            true,
        );

        let check_dump = |state: &ControllerState, partition: &str, schema: Map| {
            state.validate_dump_schema(&vec![
                ("partition_id".into(), partition.into()),
                ("field_schema".into(), Value::Map(schema)),
            ])
        };
        assert!(check_dump(&state, "missing", vec![]).is_ok());
        assert!(
            check_dump(
                &state,
                "p_dump",
                vec![
                    ("ids".into(), tensor_schema("torch.long", &[4])),
                    ("extra".into(), tensor_schema("float32", &[2])),
                ],
            )
            .is_ok()
        );
        assert!(
            check_dump(
                &state,
                "p_dump",
                vec![("ids".into(), tensor_schema("float32", &[4]))]
            )
            .unwrap_err()
            .contains("dtype mismatch")
        );
        assert!(
            check_dump(
                &state,
                "p_dump",
                vec![(
                    "ids".into(),
                    Value::Map(vec![("is_non_tensor".into(), true.into())]),
                )],
            )
            .unwrap_err()
            .contains("type mismatch")
        );

        // Clearing `ids` drops its column schema (allowing a fresh dtype later) while keeping `mask` ready.
        let cleared = state
            .kv_clear_fields(&vec![
                ("partition_id".into(), "p_dump".into()),
                ("keys".into(), strs(&["k0"])),
                ("fields".into(), strs(&["ids"])),
            ])
            .unwrap();
        assert_eq!(cleared.indexes, vec![idx]);
        assert_eq!(ready_fields(&state, "p_dump", "k0"), strs(&["mask"]));
        assert!(
            check_dump(
                &state,
                "p_dump",
                vec![("ids".into(), tensor_schema("float32", &[4]))]
            )
            .is_ok()
        );
    }
}
