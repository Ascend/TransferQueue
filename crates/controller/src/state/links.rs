// Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, BinaryHeap, HashSet};

use super::{
    ControllerState, FieldSchema, GlobalIndex, PartitionState, StateResult, global_index, strings,
};
use crate::wire::{BatchMeta, Map, field, list, text};
use rmpv::Value;

// Shared payloads live once in `PUBLIC_PARTITION` while consumer partitions store lightweight
// `(field -> public GlobalIndex)` aliases in `links`. `ever_linked` tracks public rows that have
// been referenced at least once so `mark_unlinked_public` can reclaim orphaned rows without
// sweeping newly staged rows that no consumer has linked yet.
pub(super) const PUBLIC_PARTITION: &str = "__public__";

impl PartitionState {
    fn link_fields(
        &mut self,
        index: GlobalIndex,
        public: &Self,
        source: GlobalIndex,
        fields: &[&str],
        ready: bool,
    ) {
        let row = self.rows[&index];
        for &name in fields {
            let schema = public.fields[name]
                .schema
                .as_ref()
                .expect("public field schema");
            let shape = if schema.nested {
                schema.sample_shapes.get(&source).cloned().flatten()
            } else {
                schema.shape.clone()
            };
            let schema = FieldSchema {
                dtype: schema.dtype.clone(),
                shape,
                non_tensor: schema.non_tensor,
                ..Default::default()
            };
            self.fields.entry(name.to_owned()).or_default().update(
                &[index],
                &[row],
                &self.indexes,
                schema,
                ready,
            );
            let previous = self
                .links
                .entry(index)
                .or_default()
                .insert(name.to_owned(), source);
            self.linked_by.entry(source).or_default().insert(index);
            if let Some(previous) = previous
                && previous != source
                && !self.links[&index].values().any(|&other| other == previous)
            {
                self.remove_linked_by(previous, index);
            }
        }
    }

    pub(super) fn drop_links(&mut self, index: GlobalIndex, fields: Option<&[&str]>) {
        let Some(mut row_links) = self.links.remove(&index) else {
            return;
        };
        match fields {
            None => {
                for source in row_links.into_values() {
                    self.remove_linked_by(source, index);
                }
            }
            Some(fields) => {
                for &name in fields {
                    if let Some(source) = row_links.remove(name)
                        && !row_links.values().any(|&other| other == source)
                    {
                        self.remove_linked_by(source, index);
                    }
                }
                if !row_links.is_empty() {
                    self.links.insert(index, row_links);
                }
            }
        }
    }

    fn remove_linked_by(&mut self, source: GlobalIndex, index: GlobalIndex) {
        if let Some(linkers) = self.linked_by.get_mut(&source) {
            linkers.remove(&index);
            if linkers.is_empty() {
                self.linked_by.remove(&source);
            }
        }
    }
}

impl ControllerState {
    pub(super) fn kv_link(&mut self, body: &Map) -> StateResult<()> {
        let name = text(body, "partition_id")?;
        let keys = strings(body, "keys")?;
        let public_keys = strings(body, "public_keys")?;
        let fields = strings(body, "fields")?;
        if name == PUBLIC_PARTITION {
            return Err(format!(
                "keys of partition {PUBLIC_PARTITION:?} hold shared payloads and cannot link"
            ));
        }
        if keys.is_empty() {
            return Err("keys must not be empty".into());
        }
        if keys.len() != public_keys.len() {
            return Err(format!(
                "{} public keys for {} keys",
                public_keys.len(),
                keys.len()
            ));
        }
        if keys.iter().collect::<HashSet<_>>().len() != keys.len() {
            return Err("keys must be unique".into());
        }
        let public = self.partitions.get(PUBLIC_PARTITION);
        let missing: BTreeSet<_> = public_keys
            .iter()
            .filter(|&&key| public.is_none_or(|public| !public.keys.contains_key(key)))
            .collect();
        if !missing.is_empty() {
            let keys = missing
                .into_iter()
                .take(20)
                .map(|key| format!("'{key}'"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "keys [{keys}] not found in partition '{PUBLIC_PARTITION}'"
            ));
        }
        let sources = public_keys
            .iter()
            .map(|&key| self.partitions[PUBLIC_PARTITION].keys[key])
            .collect::<Vec<_>>();
        let partition = self.partitions.get(name);
        let indexes = keys
            .iter()
            .map(|&key| partition.and_then(|p| p.keys.get(key).copied()))
            .collect::<Vec<_>>();
        self.validate_links(name, &indexes, &sources, &fields)?;
        self.register_keys(name, &keys);
        let indexes: Vec<_> = keys
            .iter()
            .map(|&key| self.partitions[name].keys[key])
            .collect();
        self.link_rows(name, &indexes, &sources, &fields);
        Ok(())
    }

    pub(super) fn link(&mut self, body: &Map) -> StateResult<BatchMeta> {
        let name = text(body, "partition_id")?;
        let fields = strings(body, "fields")?;
        let data_fields = strings(body, "data_fields")?;
        let indexes = list(body, "global_indexes")?
            .iter()
            .map(global_index)
            .collect::<StateResult<Vec<_>>>()?;
        let sources = list(body, "public_indexes")?
            .iter()
            .map(global_index)
            .collect::<StateResult<Vec<_>>>()?;
        if indexes.len() != sources.len() {
            return Err(format!(
                "{} public rows for {} rows",
                sources.len(),
                indexes.len()
            ));
        }
        if indexes.iter().collect::<HashSet<_>>().len() != indexes.len() {
            return Err("rows must be unique".into());
        }
        let optional_indexes: Vec<_> = indexes.iter().copied().map(Some).collect();
        self.validate_links(name, &optional_indexes, &sources, &fields)?;
        self.link_rows(name, &indexes, &sources, &fields);
        let partition = &self.partitions[name];
        let rows: Vec<_> = indexes.iter().map(|index| partition.rows[index]).collect();
        Ok(partition.snapshot(name, &rows, &data_fields, "force_fetch"))
    }

    fn validate_links(
        &self,
        name: &str,
        indexes: &[Option<GlobalIndex>],
        sources: &[GlobalIndex],
        fields: &[&str],
    ) -> StateResult<()> {
        if name == PUBLIC_PARTITION {
            return Err(format!(
                "rows of partition {PUBLIC_PARTITION:?} hold shared payloads and cannot link"
            ));
        }
        if indexes.is_empty() {
            return Err("rows must not be empty".into());
        }
        let public = self
            .partitions
            .get(PUBLIC_PARTITION)
            .ok_or("public partition does not exist")?;
        for source in sources {
            let row = public
                .rows
                .get(source)
                .filter(|&&row| public.active.get(row))
                .ok_or_else(|| {
                    format!("row {source} not found in partition {PUBLIC_PARTITION:?}")
                })?;
            if fields.is_empty()
                || fields.iter().any(|&name| {
                    !public
                        .fields
                        .get(name)
                        .is_some_and(|field| field.ready.get(*row))
                })
            {
                return Err(format!(
                    "public row {source} must have produced fields {fields:?}"
                ));
            }
        }
        let partition = self.partitions.get(name);
        for index in indexes.iter().flatten() {
            let row = partition
                .and_then(|partition| {
                    partition
                        .rows
                        .get(index)
                        .filter(|&&row| partition.active.get(row))
                })
                .ok_or_else(|| format!("row {index} not found in partition {name:?}"))?;
            let partition = &self.partitions[name];
            let links = partition.links.get(index);
            if fields.iter().any(|&name| {
                !links.is_some_and(|links| links.contains_key(name))
                    && partition
                        .fields
                        .get(name)
                        .is_some_and(|field| field.present.get(*row))
            }) {
                return Err(format!(
                    "rows [{index}] already store some of fields {fields:?}; clear those fields before linking them"
                ));
            }
        }
        if let Some(partition) = partition {
            for &name in fields {
                let incoming = public.fields[name]
                    .schema
                    .as_ref()
                    .expect("public field schema");
                if let Some(previous) = partition
                    .fields
                    .get(name)
                    .and_then(|field| field.schema.as_ref())
                    && let (Some(previous), Some(incoming)) = (&previous.dtype, &incoming.dtype)
                    && previous != incoming
                {
                    return Err(format!(
                        "Field '{name}' dtype mismatch: torch.{previous} != torch.{incoming}"
                    ));
                }
            }
        }
        Ok(())
    }

    fn link_rows(
        &mut self,
        name: &str,
        indexes: &[GlobalIndex],
        sources: &[GlobalIndex],
        fields: &[&str],
    ) {
        let [Some(partition), Some(public)] =
            self.partitions.get_disjoint_mut([name, PUBLIC_PARTITION])
        else {
            unreachable!("link validated both partitions")
        };
        for (&index, &source) in indexes.iter().zip(sources) {
            partition.link_fields(index, public, source, fields, true);
            public.ever_linked.insert(source);
        }
    }

    pub(super) fn check_public_not_linked(
        &self,
        indexes: &[GlobalIndex],
        fields: Option<&[&str]>,
    ) -> StateResult<()> {
        if indexes.is_empty() {
            return Ok(());
        }
        let sources: BTreeSet<_> = indexes.iter().copied().collect();
        let mut linked = BTreeSet::new();
        let mut total_linkers = 0usize;
        let mut examples = BinaryHeap::with_capacity(20);
        for (partition_name, partition) in &self.partitions {
            if partition.linked_by.is_empty() {
                continue;
            }
            for &source in &sources {
                let Some(consumers) = partition.linked_by.get(&source) else {
                    continue;
                };
                for &index in consumers {
                    let matches = fields.is_none_or(|fields| {
                        fields
                            .iter()
                            .any(|&name| partition.links[&index].get(name) == Some(&source))
                    });
                    if matches {
                        linked.insert(source);
                        total_linkers += 1;
                        let entry = match partition.index_keys.get(&index) {
                            Some(key) => format!("'{partition_name}:{key}'"),
                            None => format!("'{partition_name}:{index}'"),
                        };
                        if examples.len() < 20 {
                            examples.push(entry);
                        } else if let Some(mut max) = examples.peek_mut()
                            && entry < *max
                        {
                            *max = entry;
                        }
                    }
                }
            }
        }
        if total_linkers == 0 {
            return Ok(());
        }
        let public = &self.partitions[PUBLIC_PARTITION];
        let keys: Vec<_> = linked
            .iter()
            .map(|source| {
                public
                    .index_keys
                    .get(source)
                    .cloned()
                    .unwrap_or_else(|| source.to_string())
            })
            .collect();
        let examples = examples.into_sorted_vec().join(", ");
        Err(format!(
            "keys {keys:?} in partition '{PUBLIC_PARTITION}' are linked by keys of other partitions ({total_linkers} keys, e.g. [{examples}]); clear those keys first"
        ))
    }

    pub(super) fn unlink_public_rows(&mut self, indexes: &[GlobalIndex]) {
        let sources: HashSet<_> = indexes.iter().copied().collect();
        for partition in self.partitions.values_mut() {
            if partition.linked_by.is_empty() {
                continue;
            }
            let mut consumers = HashSet::new();
            for &source in &sources {
                if let Some(linked) = partition.linked_by.get(&source) {
                    consumers.extend(linked.iter().copied());
                }
            }
            for index in consumers {
                let fields: Vec<_> = partition.links[&index]
                    .iter()
                    .filter(|(_, source)| sources.contains(source))
                    .map(|(field, _)| field.clone())
                    .collect();
                let refs: Vec<_> = fields.iter().map(String::as_str).collect();
                let row = partition.rows[&index];
                partition.clear_fields(&[row], &refs);
            }
        }
    }

    pub(super) fn refresh_public_links(
        &mut self,
        indexes: &[GlobalIndex],
        fields: &[&str],
        ready: bool,
    ) {
        for position in 0..self.partitions.len() {
            let (name, dest) = self
                .partitions
                .get_index(position)
                .expect("partition position");
            if name == PUBLIC_PARTITION || dest.linked_by.is_empty() {
                continue;
            }
            let updates: Vec<_> = indexes
                .iter()
                .filter_map(|source| {
                    dest.linked_by
                        .get(source)
                        .map(|consumers| (*source, consumers))
                })
                .flat_map(|(source, consumers)| {
                    consumers.iter().flat_map(move |&index| {
                        fields
                            .iter()
                            .copied()
                            .filter(move |&field| dest.links[&index].get(field) == Some(&source))
                            .map(move |field| (index, source, field))
                    })
                })
                .collect();
            if updates.is_empty() {
                continue;
            }
            let name = name.clone();
            let [Some(dest), Some(public)] = self
                .partitions
                .get_disjoint_mut([name.as_str(), PUBLIC_PARTITION])
            else {
                unreachable!("refresh holds an existing public partition")
            };
            for (index, source, field) in updates {
                dest.link_fields(index, public, source, &[field], ready);
            }
        }
    }

    pub(super) fn mark_unlinked_public_clearing(
        &mut self,
        body: &Map,
    ) -> StateResult<(BatchMeta, Vec<Value>)> {
        let stale_keys = if field(body, "stale_keys").is_some_and(|value| !value.is_nil()) {
            Some(strings(body, "stale_keys")?)
        } else {
            None
        };
        let (metadata, keys) = self.mark_unlinked_public(stale_keys.as_deref());
        Ok((metadata, keys.into_iter().map(Value::from).collect()))
    }

    pub(crate) fn mark_unlinked_public(
        &mut self,
        stale_keys: Option<&[&str]>,
    ) -> (BatchMeta, Vec<String>) {
        let Some(public) = self.partitions.get(PUBLIC_PARTITION) else {
            return (BatchMeta::default(), Vec::new());
        };
        let mut candidates: BTreeSet<_> = public.ever_linked.iter().copied().collect();
        if let Some(stale_keys) = stale_keys {
            candidates.extend(
                stale_keys
                    .iter()
                    .filter_map(|&key| public.keys.get(key).copied()),
            );
        }
        let indexes: Vec<_> = candidates
            .into_iter()
            .filter(|index| {
                !self
                    .partitions
                    .values()
                    .any(|partition| partition.linked_by.contains_key(index))
            })
            .collect();
        if indexes.is_empty() {
            return (BatchMeta::default(), Vec::new());
        }
        let public = self
            .partitions
            .get_mut(PUBLIC_PARTITION)
            .expect("public partition exists");
        let rows: Vec<_> = indexes.iter().map(|index| public.rows[index]).collect();
        let fields = public.fields_ready_in_all(&rows);
        let metadata = public.snapshot(PUBLIC_PARTITION, &rows, &fields, "force_fetch");
        public.mark_clearing(&indexes);
        let mut keys: Vec<_> = indexes
            .iter()
            .filter_map(|index| public.index_keys.get(index).cloned())
            .collect();
        keys.sort_unstable();
        (metadata, keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::Request;

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

    fn notify(
        state: &mut ControllerState,
        partition: &str,
        indexes: &[u64],
        schema: Map,
        ready: bool,
    ) {
        state
            .execute(&Request {
                request_type: "NOTIFY_DATA_UPDATE".into(),
                sender_id: "test".into(),
                body: vec![
                    ("partition_id".into(), partition.into()),
                    (
                        "global_indexes".into(),
                        Value::Array(indexes.iter().copied().map(Value::from).collect()),
                    ),
                    ("field_schema".into(), Value::Map(schema)),
                    ("ready".into(), ready.into()),
                ],
            })
            .unwrap();
    }

    fn kv_link(
        state: &mut ControllerState,
        partition: &str,
        keys: &[&str],
        public_keys: &[&str],
        fields: &[&str],
    ) -> StateResult<()> {
        state.kv_link(&vec![
            ("partition_id".into(), partition.into()),
            ("keys".into(), strs(keys)),
            ("public_keys".into(), strs(public_keys)),
            ("fields".into(), strs(fields)),
        ])
    }

    #[test]
    fn kv_link_validation_and_public_reference_protection() {
        let mut state = ControllerState::new(1, 1);
        let pub_meta = state
            .kv_retrieve_meta(PUBLIC_PARTITION, &["pub_ready", "pub_unready"], true)
            .unwrap();
        notify(
            &mut state,
            PUBLIC_PARTITION,
            &[pub_meta.indexes[0]],
            vec![("prompt".into(), tensor_schema("int64", &[4]))],
            true,
        );
        notify(
            &mut state,
            PUBLIC_PARTITION,
            &[pub_meta.indexes[1]],
            vec![("prompt".into(), tensor_schema("int64", &[4]))],
            false,
        );

        assert!(
            kv_link(
                &mut state,
                PUBLIC_PARTITION,
                &["x"],
                &["pub_ready"],
                &["prompt"]
            )
            .is_err()
        );
        assert!(
            kv_link(&mut state, "train", &["s0"], &["missing"], &["prompt"])
                .err()
                .unwrap()
                .contains("not found")
        );
        assert!(
            kv_link(&mut state, "train", &["s0"], &["pub_unready"], &["prompt"])
                .err()
                .unwrap()
                .contains("must have produced fields")
        );

        let conflict = state
            .kv_retrieve_meta("conflict_p", &["priv", "other"], true)
            .unwrap();
        notify(
            &mut state,
            "conflict_p",
            &[conflict.indexes[0]],
            vec![("prompt".into(), tensor_schema("float32", &[4]))],
            true,
        );
        assert!(
            kv_link(
                &mut state,
                "conflict_p",
                &["priv"],
                &["pub_ready"],
                &["prompt"]
            )
            .err()
            .unwrap()
            .contains("already store")
        );
        assert!(
            kv_link(
                &mut state,
                "conflict_p",
                &["other"],
                &["pub_ready"],
                &["prompt"]
            )
            .err()
            .unwrap()
            .contains("dtype mismatch")
        );

        kv_link(&mut state, "train", &["s0"], &["pub_ready"], &["prompt"]).unwrap();
        assert!(
            state
                .kv_retrieve_meta(PUBLIC_PARTITION, &["pub_ready"], true)
                .err()
                .unwrap()
                .contains("are linked by keys")
        );
        assert!(
            state
                .kv_clear_fields(&vec![
                    ("partition_id".into(), PUBLIC_PARTITION.into()),
                    ("keys".into(), strs(&["pub_ready"])),
                    ("fields".into(), strs(&["prompt"])),
                ])
                .err()
                .unwrap()
                .contains("are linked by keys")
        );
        assert!(
            state
                .kv_prepare_update(&vec![
                    ("partition_id".into(), "train".into()),
                    ("keys".into(), strs(&["s0"])),
                    ("fields".into(), strs(&["prompt"])),
                ])
                .err()
                .unwrap()
                .contains("cannot merge into fields linked")
        );

        let alt_idx = state
            .kv_retrieve_meta(PUBLIC_PARTITION, &["pub_alt"], true)
            .unwrap()
            .indexes[0];
        notify(
            &mut state,
            PUBLIC_PARTITION,
            &[alt_idx],
            vec![("prompt".into(), tensor_schema("int64", &[4]))],
            true,
        );
        kv_link(&mut state, "train", &["s0"], &["pub_alt"], &["prompt"]).unwrap();
        assert!(
            state
                .kv_retrieve_meta(PUBLIC_PARTITION, &["pub_ready"], true)
                .is_ok()
        );
    }

    #[test]
    fn public_link_refresh_unlink_and_orphan_reclamation() {
        let mut state = ControllerState::new(1, 1);
        let pub_meta = state
            .kv_retrieve_meta(PUBLIC_PARTITION, &["pub_used", "pub_staged"], true)
            .unwrap();
        let used_idx = pub_meta.indexes[0];
        notify(
            &mut state,
            PUBLIC_PARTITION,
            &pub_meta.indexes,
            vec![("prompt".into(), tensor_schema("int64", &[4]))],
            true,
        );
        kv_link(&mut state, "train", &["s0"], &["pub_used"], &["prompt"]).unwrap();

        // Unfinalized public rewrite hides consumer readiness; finalizing with jagged shape propagates it.
        let jagged_prompt = vec![(
            "prompt".into(),
            Value::Map(vec![
                ("dtype".into(), "int64".into()),
                ("is_nested".into(), true.into()),
                ("is_non_tensor".into(), false.into()),
                (
                    "per_sample_shapes".into(),
                    Value::Map(vec![(used_idx.into(), Value::Array(vec![6.into()]))]),
                ),
            ]),
        )];
        notify(
            &mut state,
            PUBLIC_PARTITION,
            &[used_idx],
            jagged_prompt.clone(),
            false,
        );
        assert_eq!(
            state
                .kv_retrieve_meta("train", &["s0"], false)
                .unwrap()
                .status,
            vec![0]
        );

        notify(
            &mut state,
            PUBLIC_PARTITION,
            &[used_idx],
            jagged_prompt,
            true,
        );
        let refreshed = state.kv_retrieve_meta("train", &["s0"], false).unwrap();
        assert_eq!(refreshed.status, vec![1]);
        let (_, prompt_schema) = refreshed
            .schema
            .iter()
            .find(|(name, _)| name == "prompt")
            .unwrap();
        assert_eq!(
            prompt_schema.sample_shapes.as_ref().unwrap(),
            &vec![Some(vec![6])]
        );

        // `mark_unlinked_public` skips active links and never-linked `pub_staged`, reclaims orphaned `pub_used`
        // once `train` clears, and clears `pub_staged` only when explicitly passed via `stale_keys`.
        assert!(state.mark_unlinked_public(None).1.is_empty());
        state.clear_partition("train");
        let (reclaimed, keys) = state.mark_unlinked_public(None);
        assert_eq!(keys, vec!["pub_used".to_string()]);
        assert_eq!(reclaimed.indexes, vec![used_idx]);
        state
            .clear_meta_rows(&[used_idx], &[PUBLIC_PARTITION.to_string()])
            .unwrap();
        assert!(state.mark_unlinked_public(None).1.is_empty());
        assert_eq!(
            state.mark_unlinked_public(Some(&["pub_staged"])).1,
            vec!["pub_staged".to_string()]
        );
    }
}
