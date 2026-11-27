// Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};

use indexmap::IndexMap;
use rmpv::Value;
use serde::{Deserialize, Serialize};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

mod checkpoint;
mod kv;
mod links;
mod metrics;

use links::PUBLIC_PARTITION;

use crate::wire::{BatchMeta, Body, FieldSchemaWire, Map, RawMsgpack, Request, field, list, text};

type GlobalIndex = u64;
type LocalRow = usize;
type StateResult<T> = Result<T, String>;

const UNASSIGNED_SEQ: u64 = u64::MAX;

// Dense bitset indexed by partition-local row (`row / 64`, bit `row % 64`) rather than sparse
// `GlobalIndex`, enabling 64-way bitwise readiness/consumption scans without tensor allocations.
#[derive(Default, Deserialize, Serialize)]
struct Bitmap(Vec<u64>);

impl Bitmap {
    fn word(&self, index: usize) -> u64 {
        self.0.get(index).copied().unwrap_or(0)
    }

    fn get(&self, row: LocalRow) -> bool {
        self.word(row / 64) & (1 << (row % 64)) != 0
    }

    fn highest_set(&self) -> Option<LocalRow> {
        self.0
            .iter()
            .rposition(|&word| word != 0)
            .map(|word_idx| word_idx * 64 + 63 - self.0[word_idx].leading_zeros() as usize)
    }

    fn swap_remove(&mut self, row: LocalRow, last: LocalRow) {
        self.set(row, self.get(last));
        self.set(last, false);
        self.0.truncate(last.div_ceil(64));
    }

    fn set(&mut self, row: LocalRow, value: bool) {
        let word = row / 64;
        if value && word >= self.0.len() {
            self.0.resize(word + 1, 0);
        }
        if let Some(word) = self.0.get_mut(word) {
            let bit = 1 << (row % 64);
            *word = if value { *word | bit } else { *word & !bit };
        }
    }
}

// `SimpleStorageManager` routes payloads by `global_index % num_storage_units`, so choosing an
// index's remainder chooses its storage unit. Keep per-unit FIFO reuse pools and steer each new
// allocation to this partition's least-loaded unit (preferring recycled slots, then round-robin
// `self.next`, banking skipped counter values into their own unit's pool).
#[derive(Deserialize, Serialize)]
struct IndexAllocator {
    next: GlobalIndex,
    reusable: Vec<VecDeque<GlobalIndex>>,
}

impl IndexAllocator {
    fn allocate(&mut self, loads: &mut [usize]) -> GlobalIndex {
        let units = loads.len() as u64;
        let next_unit = (self.next % units) as usize;
        let unit = (0..loads.len())
            .min_by_key(|&unit| {
                (
                    loads[unit],
                    self.reusable[unit].is_empty(),
                    unit != next_unit,
                )
            })
            .expect("at least one storage unit");
        loads[unit] += 1;
        if let Some(index) = self.reusable[unit].pop_front() {
            return index;
        }
        while self.next % units != unit as u64 {
            self.reusable[(self.next % units) as usize].push_back(self.next);
            self.next += 1;
        }
        let index = self.next;
        self.next += 1;
        index
    }

    fn release(&mut self, index: GlobalIndex) {
        let unit = (index % self.reusable.len() as u64) as usize;
        self.reusable[unit].push_back(index);
    }
}

#[derive(Clone, Default, Deserialize, Serialize)]
struct FieldSchema {
    dtype: Option<String>,
    shape: Option<Vec<u64>>,
    nested: bool,
    non_tensor: bool,
    sample_shapes: HashMap<GlobalIndex, Option<Vec<u64>>>,
}

impl FieldSchema {
    fn parse(value: &Value, name: &str, indexes: &[GlobalIndex]) -> StateResult<Self> {
        let Value::Map(fields) = value else {
            return Err("field schema must be a map".into());
        };
        let dtype = match field(fields, "dtype") {
            None | Some(Value::Nil) => None,
            Some(value) => Some(parse_dtype(value, name)?.to_owned()),
        };
        let shape = parse_shape(field(fields, "shape").unwrap_or(&Value::Nil))?;
        let nested = flag(fields, "is_nested", false)?;
        let non_tensor = flag(fields, "is_non_tensor", false)?;
        let sample_shapes = match field(fields, "per_sample_shapes") {
            Some(Value::Map(shapes)) => shapes
                .iter()
                .map(|(index, shape)| Ok((global_index(index)?, parse_shape(shape)?)))
                .collect::<StateResult<_>>()?,
            Some(Value::Array(shapes)) if shapes.len() == indexes.len() => indexes
                .iter()
                .copied()
                .zip(shapes)
                .map(|(index, shape)| Ok((index, parse_shape(shape)?)))
                .collect::<StateResult<_>>()?,
            None | Some(Value::Nil) if !nested => HashMap::new(),
            _ => return Err("invalid per_sample_shapes".into()),
        };
        if !nested && !non_tensor && shape.is_none() {
            return Err("regular tensor schema requires a shape".into());
        }
        Ok(Self {
            dtype,
            shape,
            nested,
            non_tensor,
            sample_shapes,
        })
    }

    fn wire_snapshot(&self, indexes: &[GlobalIndex]) -> FieldSchemaWire {
        let sample_shapes = (self.nested && !self.sample_shapes.is_empty()).then(|| {
            indexes
                .iter()
                .map(|index| self.sample_shapes.get(index).cloned().flatten())
                .collect()
        });
        FieldSchemaWire {
            dtype: self.dtype.clone(),
            shape: self.shape.clone(),
            nested: self.nested,
            non_tensor: self.non_tensor,
            sample_shapes,
        }
    }
}

fn parse_shape(value: &Value) -> StateResult<Option<Vec<u64>>> {
    match value {
        Value::Nil => Ok(None),
        Value::Array(dims) => dims
            .iter()
            .map(global_index)
            .collect::<StateResult<_>>()
            .map(Some),
        _ => Err("shape must be an array".into()),
    }
}

// `present` tracks rows holding any stored value (including unfinalized `kv_update` writes), while
// `ready` gates consumer visibility. Heterogeneous shapes promote a column to `nested` (`sample_shapes`)
// and collapse back to a uniform `shape` once conflicting rows are cleared.
#[derive(Default, Deserialize, Serialize)]
struct FieldState {
    schema: Option<FieldSchema>,
    present: Bitmap,
    ready: Bitmap,
}

impl FieldState {
    fn remove_empty_schema(&mut self) {
        let present: usize = self
            .present
            .0
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum();
        if present == 0 {
            self.schema = None;
        } else if let Some(schema) = &mut self.schema
            && schema.nested
            && !schema.non_tensor
            && schema.sample_shapes.len() == present
            && let Some(Some(shape)) = schema.sample_shapes.values().next()
            && schema
                .sample_shapes
                .values()
                .all(|value| value.as_ref() == Some(shape))
        {
            schema.shape = Some(shape.clone());
            schema.nested = false;
            schema.sample_shapes.clear();
        }
    }

    fn update(
        &mut self,
        indexes: &[GlobalIndex],
        rows: &[LocalRow],
        all_indexes: &[GlobalIndex],
        incoming: FieldSchema,
        ready: bool,
    ) {
        if let Some(schema) = &mut self.schema {
            if schema.dtype.is_none() {
                schema.dtype = incoming.dtype;
            }
            let shape_changed = !incoming.non_tensor && schema.shape != incoming.shape;
            if !schema.nested
                && (incoming.nested || shape_changed)
                && let Some(old_shape) = schema.shape.take()
            {
                for (row, index) in all_indexes.iter().enumerate() {
                    if self.present.get(row) {
                        schema.sample_shapes.insert(*index, Some(old_shape.clone()));
                    }
                }
                schema.nested = true;
            }
            if incoming.nested {
                schema.sample_shapes.extend(incoming.sample_shapes);
            } else if schema.nested && !incoming.non_tensor {
                for index in indexes {
                    schema.sample_shapes.insert(*index, incoming.shape.clone());
                }
            }
        } else if !indexes.is_empty() {
            self.schema = Some(incoming);
        }
        for &row in rows {
            self.present.set(row, true);
            self.ready.set(row, ready);
        }
    }
}

// Maps cluster-wide `GlobalIndex` slots onto dense partition-local rows (`0..indexes.len()`) so
// bitmaps stay compact even when global indexes are sparse. Each activated sample receives a
// strictly monotonic `seqs[row]` ordinal so built-in `fetch` always consumes in true insertion FIFO
// order rather than recycled physical `GlobalIndex` order.
#[derive(Deserialize, Serialize)]
struct PartitionState {
    created_at: f64,
    indexes: Vec<GlobalIndex>,
    seqs: Vec<u64>,
    next_seq: u64,
    #[serde(skip)]
    seq_sorted: bool,
    rows: HashMap<GlobalIndex, LocalRow>,
    active: Bitmap,
    reserved: BTreeSet<GlobalIndex>,
    #[serde(skip)]
    loads: Vec<usize>,
    fields: IndexMap<String, FieldState>,
    keys: IndexMap<String, GlobalIndex>,
    index_keys: HashMap<GlobalIndex, String>,
    consumed: HashMap<String, Bitmap>,
    #[serde(skip)]
    consumed_cursor: HashMap<String, usize>,
    backend_meta: HashMap<GlobalIndex, HashMap<String, RawMsgpack>>,
    custom_meta: HashMap<GlobalIndex, RawMsgpack>,
    links: HashMap<GlobalIndex, HashMap<String, GlobalIndex>>,
    #[serde(skip)]
    linked_by: HashMap<GlobalIndex, HashSet<GlobalIndex>>,
    ever_linked: HashSet<GlobalIndex>,
}

impl PartitionState {
    fn new(units: usize) -> Self {
        Self {
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            indexes: Vec::new(),
            seqs: Vec::new(),
            next_seq: 0,
            seq_sorted: true,
            rows: HashMap::new(),
            active: Bitmap::default(),
            reserved: BTreeSet::new(),
            loads: vec![0; units],
            fields: IndexMap::new(),
            keys: IndexMap::new(),
            index_keys: HashMap::new(),
            consumed: HashMap::new(),
            consumed_cursor: HashMap::new(),
            backend_meta: HashMap::new(),
            custom_meta: HashMap::new(),
            links: HashMap::new(),
            linked_by: HashMap::new(),
            ever_linked: HashSet::new(),
        }
    }

    fn add_row(&mut self, allocator: &mut IndexAllocator) -> LocalRow {
        let index = allocator.allocate(&mut self.loads);
        let row = self.indexes.len();
        self.indexes.push(index);
        self.seqs.push(UNASSIGNED_SEQ);
        self.rows.insert(index, row);
        row
    }

    fn activate_row(&mut self, row: LocalRow) -> bool {
        if self.active.get(row) {
            return false;
        }
        if self.seq_sorted
            && self
                .active
                .highest_set()
                .is_some_and(|highest| row < highest)
        {
            self.seq_sorted = false;
        }
        self.active.set(row, true);
        self.seqs[row] = self.next_seq;
        self.next_seq += 1;
        true
    }

    fn insert(&mut self, count: usize, allocator: &mut IndexAllocator) -> Vec<LocalRow> {
        (0..count)
            .map(|_| {
                let row = match self.reserved.pop_first() {
                    Some(index) => self.rows[&index],
                    None => self.add_row(allocator),
                };
                self.activate_row(row);
                row
            })
            .collect()
    }

    fn fetch(&mut self, fields: &[&str], task: &str, count: usize) -> Vec<LocalRow> {
        let Some(ready) = fields
            .iter()
            .map(|name| self.fields.get(*name).map(|field| &field.ready))
            .collect::<Option<Vec<_>>>()
        else {
            return Vec::new();
        };
        let consumed = self.consumed.entry(task.to_owned()).or_default();
        if count == 0 || count > self.indexes.len() {
            return Vec::new();
        }
        let mut cursor = self.consumed_cursor.get(task).copied().unwrap_or(0);
        while cursor < self.active.0.len() && consumed.word(cursor) == u64::MAX {
            cursor += 1;
        }
        let start_word = cursor;

        let rows = if self.seq_sorted {
            // Fast path: active rows are strictly ordered by insertion sequence, so scanning from
            // `start_word` can stop immediately once `count` ready samples are collected.
            let mut selected = Vec::with_capacity(count);
            for word in start_word..self.active.0.len() {
                let mut candidates = self.active.word(word) & !consumed.word(word);
                for field in &ready {
                    candidates &= field.word(word);
                }
                while candidates != 0 {
                    selected.push(word * 64 + candidates.trailing_zeros() as usize);
                    if selected.len() == count {
                        break;
                    }
                    candidates &= candidates - 1;
                }
                if selected.len() == count {
                    break;
                }
            }
            if selected.len() < count {
                self.consumed_cursor.insert(task.to_owned(), cursor);
                return Vec::new();
            }
            selected
        } else {
            // General path after mid-partition `swap_remove`: maintain the lowest `count` insertion
            // sequence numbers in a bounded max-heap instead of materializing all ready candidates.
            let mut total_candidates = 0usize;
            let mut smallest = BinaryHeap::with_capacity(count + 1);
            for word in start_word..self.active.0.len() {
                let mut candidates = self.active.word(word) & !consumed.word(word);
                for field in &ready {
                    candidates &= field.word(word);
                }
                while candidates != 0 {
                    let row = word * 64 + candidates.trailing_zeros() as usize;
                    candidates &= candidates - 1;
                    total_candidates += 1;
                    let seq = self.seqs[row];
                    if smallest.len() < count {
                        smallest.push((seq, row));
                    } else if let Some(&(max_seq, _)) = smallest.peek()
                        && seq < max_seq
                    {
                        smallest.pop();
                        smallest.push((seq, row));
                    }
                }
            }
            if total_candidates < count {
                self.consumed_cursor.insert(task.to_owned(), cursor);
                return Vec::new();
            }
            smallest
                .into_sorted_vec()
                .into_iter()
                .map(|(_, row)| row)
                .collect()
        };

        for &row in &rows {
            consumed.set(row, true);
        }
        while cursor < self.active.0.len() && consumed.word(cursor) == u64::MAX {
            cursor += 1;
        }
        self.consumed_cursor.insert(task.to_owned(), cursor);
        rows
    }

    fn snapshot(
        &self,
        partition_id: &str,
        rows: &[LocalRow],
        fields: &[&str],
        mode: &str,
    ) -> BatchMeta {
        let indexes: Vec<_> = rows.iter().map(|row| self.indexes[*row]).collect();
        let mut schema = Vec::with_capacity(fields.len());
        let mut selected_fields = Vec::with_capacity(fields.len());
        for &name in fields {
            let field = self.fields.get(name);
            if let Some(value) = field.and_then(|field| field.schema.as_ref()) {
                schema.push((name.to_owned(), value.wire_snapshot(&indexes)));
            } else if mode == "insert" {
                schema.push((
                    name.to_owned(),
                    FieldSchema::default().wire_snapshot(&indexes),
                ));
            }
            if let Some(field) = field {
                selected_fields.push(field);
            }
        }
        let status = match mode {
            "fetch" => vec![1; indexes.len()],
            "insert" => vec![0; indexes.len()],
            _ => rows
                .iter()
                .map(|&row| {
                    u8::from(
                        !fields.is_empty()
                            && selected_fields.len() == fields.len()
                            && selected_fields.iter().all(|field| field.ready.get(row)),
                    )
                })
                .collect(),
        };
        let backend_meta = if self.backend_meta.is_empty() {
            Vec::new()
        } else {
            indexes
                .iter()
                .map(|index| {
                    let Some(values) = self.backend_meta.get(index) else {
                        return Vec::new();
                    };
                    fields
                        .iter()
                        .filter_map(|&name| {
                            values.get(name).map(|raw| (name.to_owned(), raw.clone()))
                        })
                        .collect()
                })
                .collect()
        };
        let custom_meta = if self.custom_meta.is_empty() {
            Vec::new()
        } else {
            indexes
                .iter()
                .map(|index| self.custom_meta.get(index).cloned())
                .collect()
        };
        let links = if self.links.is_empty() {
            Vec::new()
        } else {
            indexes
                .iter()
                .map(|index| {
                    let Some(row_links) = self.links.get(index) else {
                        return Vec::new();
                    };
                    fields
                        .iter()
                        .filter_map(|&name| {
                            row_links.get(name).map(|&source| (name.to_owned(), source))
                        })
                        .collect()
                })
                .collect()
        };
        BatchMeta {
            partition_id: partition_id.to_owned(),
            indexes,
            schema,
            status,
            backend_meta,
            custom_meta,
            links,
        }
    }

    fn sorted_rows(&self) -> Vec<LocalRow> {
        let mut rows: Vec<_> = (0..self.indexes.len()).collect();
        rows.sort_unstable_by_key(|&row| self.indexes[row]);
        rows
    }

    fn mark_clearing(&mut self, indexes: &[GlobalIndex]) {
        for row in indexes.iter().filter_map(|index| self.rows.get(index)) {
            for field in self.fields.values_mut() {
                field.ready.set(*row, false);
            }
        }
    }

    fn clear_data(&mut self, indexes: &[GlobalIndex], allocator: &mut IndexAllocator) {
        for &index in indexes {
            let Some(&row) = self.rows.get(&index) else {
                continue;
            };
            if !self.active.get(row) {
                continue;
            }
            let last = self.indexes.len() - 1;
            self.rows.remove(&index);
            self.indexes.swap_remove(row);
            self.seqs.swap_remove(row);
            if row != last {
                self.rows.insert(self.indexes[row], row);
            }
            self.active.swap_remove(row, last);
            if row != last
                && self.active.get(row)
                && self
                    .active
                    .highest_set()
                    .is_some_and(|highest| highest > row)
            {
                self.seq_sorted = false;
            }
            for consumed in self.consumed.values_mut() {
                consumed.swap_remove(row, last);
            }
            for cursor in self.consumed_cursor.values_mut() {
                *cursor = (*cursor).min(row / 64);
            }
            for field in self.fields.values_mut() {
                field.ready.swap_remove(row, last);
                field.present.swap_remove(row, last);
                if let Some(schema) = &mut field.schema {
                    schema.sample_shapes.remove(&index);
                }
            }
            self.custom_meta.remove(&index);
            self.backend_meta.remove(&index);
            if let Some(key) = self.index_keys.remove(&index) {
                self.keys.swap_remove(&key);
            }
            self.drop_links(index, None);
            self.ever_linked.remove(&index);
            let unit = (index % self.loads.len() as u64) as usize;
            self.loads[unit] = self.loads[unit].saturating_sub(1);
            allocator.release(index);
        }
        if self.active.highest_set().is_none() {
            self.next_seq = 0;
            self.seq_sorted = true;
            self.consumed_cursor.clear();
        }
        for field in self.fields.values_mut() {
            field.remove_empty_schema();
        }
    }

    fn update_production_status(
        &mut self,
        indexes: &[GlobalIndex],
        fields: &[(Value, Value)],
        backend_meta: Option<&Value>,
        ready: bool,
    ) -> StateResult<()> {
        let mut rows = Vec::with_capacity(indexes.len());
        for index in indexes {
            rows.push(
                *self
                    .rows
                    .get(index)
                    .ok_or_else(|| format!("unallocated global index {index}"))?,
            );
        }
        let mut schemas = Vec::new();
        for (name, value) in fields {
            let name = name.as_str().ok_or("field names must be strings")?;
            let incoming = FieldSchema::parse(value, name, indexes)?;
            if let Some(previous) = self
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
            schemas.push((name, incoming));
        }
        let backend_meta = parse_backend_meta(backend_meta)?;
        let names: Vec<_> = schemas.iter().map(|&(name, _)| name).collect();
        for &index in indexes {
            self.drop_links(index, Some(&names));
        }

        for (name, schema) in schemas {
            self.fields.entry(name.to_owned()).or_default().update(
                indexes,
                &rows,
                &self.indexes,
                schema,
                ready,
            );
        }
        for (&index, &row) in indexes.iter().zip(&rows) {
            if self.activate_row(row) {
                self.reserved.remove(&index);
            }
        }
        if !indexes.is_empty() {
            for (index, values) in backend_meta {
                self.backend_meta.entry(index).or_default().extend(values);
            }
        }
        Ok(())
    }
}

pub(super) fn parse_dtype(value: &Value, field_name: &str) -> StateResult<&'static str> {
    let raw = value.as_str().ok_or("dtype must be a string")?;
    let name = raw.strip_prefix("torch.").unwrap_or(raw);
    Ok(match name {
        "bfloat16" => "bfloat16",
        "bits16" => "bits16",
        "bits1x8" => "bits1x8",
        "bits2x4" => "bits2x4",
        "bits4x2" => "bits4x2",
        "bits8" => "bits8",
        "bool" => "bool",
        "cdouble" | "complex128" => "complex128",
        "chalf" | "complex32" => "complex32",
        "cfloat" | "complex64" => "complex64",
        "half" | "float16" => "float16",
        "float" | "float32" => "float32",
        "float4_e2m1fn_x2" => "float4_e2m1fn_x2",
        "double" | "float64" => "float64",
        "float8_e4m3fn" => "float8_e4m3fn",
        "float8_e4m3fnuz" => "float8_e4m3fnuz",
        "float8_e5m2" => "float8_e5m2",
        "float8_e5m2fnuz" => "float8_e5m2fnuz",
        "float8_e8m0fnu" => "float8_e8m0fnu",
        "int1" => "int1",
        "short" | "int16" => "int16",
        "int2" => "int2",
        "int3" => "int3",
        "int" | "int32" => "int32",
        "int4" => "int4",
        "int5" => "int5",
        "int6" => "int6",
        "long" | "int64" => "int64",
        "int7" => "int7",
        "int8" => "int8",
        "qint32" => "qint32",
        "qint8" => "qint8",
        "quint2x4" => "quint2x4",
        "quint4x2" => "quint4x2",
        "quint8" => "quint8",
        "bit" | "uint1" => "uint1",
        "uint16" => "uint16",
        "uint2" => "uint2",
        "uint3" => "uint3",
        "uint32" => "uint32",
        "uint4" => "uint4",
        "uint5" => "uint5",
        "uint6" => "uint6",
        "uint64" => "uint64",
        "uint7" => "uint7",
        "uint8" => "uint8",
        _ => {
            return Err(format!(
                "Invalid Torch dtype {raw:?} for field {field_name:?}"
            ));
        }
    })
}

fn parse_backend_meta(
    value: Option<&Value>,
) -> StateResult<Vec<(GlobalIndex, HashMap<String, RawMsgpack>)>> {
    let Some(value) = value.filter(|value| !value.is_nil()) else {
        return Ok(Vec::new());
    };
    value
        .as_map()
        .ok_or("custom_backend_meta must be a map")?
        .iter()
        .map(|(index, values)| {
            let values = values
                .as_map()
                .ok_or("backend metadata for a row must be a map")?
                .iter()
                .map(|(name, value)| {
                    Ok((
                        name.as_str()
                            .ok_or("backend metadata field must be a string")?
                            .to_owned(),
                        RawMsgpack::encode(value)?,
                    ))
                })
                .collect::<StateResult<_>>()?;
            Ok((global_index(index)?, values))
        })
        .collect()
}

fn global_index(value: &Value) -> StateResult<GlobalIndex> {
    value
        .as_u64()
        .ok_or_else(|| "expected a nonnegative integer".into())
}

fn flag(fields: &Map, key: &str, default: bool) -> StateResult<bool> {
    field(fields, key).map_or(Ok(default), |value| {
        value
            .as_bool()
            .ok_or_else(|| format!("{key} must be a boolean"))
    })
}

#[derive(Deserialize, Serialize)]
pub(crate) struct ControllerState {
    partitions: IndexMap<String, PartitionState>,
    allocator: IndexAllocator,
    preallocate: usize,
    #[serde(skip)]
    metrics: Option<metrics::Metrics>,
}

impl ControllerState {
    pub(crate) fn new(units: usize, preallocate: usize) -> Self {
        Self {
            partitions: IndexMap::new(),
            allocator: IndexAllocator {
                next: 0,
                reusable: vec![VecDeque::new(); units],
            },
            preallocate,
            metrics: None,
        }
    }

    pub(crate) fn num_storage_units(&self) -> usize {
        self.allocator.reusable.len()
    }

    pub(crate) fn set_num_storage_units(&mut self, units: usize) -> StateResult<()> {
        if units == 0 {
            return Err("Controller requires at least one storage unit".into());
        }
        let mut reusable = vec![VecDeque::new(); units];
        for index in self.allocator.reusable.drain(..).flatten() {
            reusable[(index % units as u64) as usize].push_back(index);
        }
        self.allocator.reusable = reusable;
        for partition in self.partitions.values_mut() {
            partition.loads = vec![0; units];
            for &index in &partition.indexes {
                partition.loads[(index % units as u64) as usize] += 1;
            }
        }
        Ok(())
    }

    pub(crate) fn execute(&mut self, request: &Request) -> StateResult<(&'static str, Body)> {
        let started = (self.metrics.is_some()
            && !matches!(
                request.request_type.as_str(),
                "SAVE_CONTROLLER_CHECKPOINT" | "LOAD_CONTROLLER_CHECKPOINT"
            ))
        .then(Instant::now);
        let result = self.apply(request);
        if let Some(started) = started
            && let Some(metrics) = &mut self.metrics
        {
            metrics.record(
                request,
                result.as_ref().ok().map(|(_, body)| body),
                started.elapsed(),
            );
        }
        result
    }

    pub(crate) fn list_partitions(&self) -> Vec<String> {
        self.partitions.keys().cloned().collect()
    }

    pub(crate) fn partition_index_range(&self, name: &str) -> Vec<GlobalIndex> {
        self.partitions
            .get(name)
            .map_or_else(Vec::new, |partition| partition.indexes.clone())
    }

    pub(crate) fn enable_metrics(&mut self, buckets: Vec<f64>) {
        self.metrics
            .get_or_insert_with(|| metrics::Metrics::new(buckets));
    }

    fn apply(&mut self, request: &Request) -> StateResult<(&'static str, Body)> {
        let body = &request.body;
        match request.request_type.as_str() {
            "GET_META" => Ok((
                "GET_META_RESPONSE",
                Body::Metadata(self.get_meta(body)?, Vec::new()),
            )),
            "NOTIFY_DATA_UPDATE" => Ok((
                "NOTIFY_DATA_UPDATE_ACK",
                Body::Fields(vec![(
                    "success".into(),
                    self.update_production_status(body)?.into(),
                )]),
            )),
            "GET_LIST_PARTITIONS" => Ok((
                "LIST_PARTITIONS_RESPONSE",
                Body::Fields(vec![(
                    "partition_ids".into(),
                    Value::Array(
                        self.partitions
                            .keys()
                            .map(|name| name.as_str().into())
                            .collect(),
                    ),
                )]),
            )),
            "GET_PARTITION_META" => {
                let name = text(body, "partition_id")?;
                let metadata = match self.partitions.get(name) {
                    Some(partition) => {
                        let fields: Vec<_> = partition.fields.keys().map(String::as_str).collect();
                        Body::Metadata(
                            partition.snapshot(
                                name,
                                &partition.sorted_rows(),
                                &fields,
                                "force_fetch",
                            ),
                            Vec::new(),
                        )
                    }
                    None => Body::Fields(vec![("metadata".into(), Value::Nil)]),
                };
                Ok(("GET_PARTITION_META_RESPONSE", metadata))
            }
            "GET_PRODUCTION" => Ok((
                "PRODUCTION_RESPONSE",
                Body::Fields(self.get_production_status(body)?),
            )),
            "GET_CONSUMPTION" => Ok((
                "CONSUMPTION_RESPONSE",
                Body::Fields(self.get_consumption_status(body)?),
            )),
            "SET_CUSTOM_META" => {
                let partitions = field(body, "partition_custom_meta")
                    .and_then(Value::as_map)
                    .ok_or("partition_custom_meta must be a map")?;
                for (name, meta) in partitions {
                    let name = name.as_str().ok_or("partition name must be a string")?;
                    if let Some(partition) = self.partitions.get_mut(name) {
                        let rows = meta
                            .as_map()
                            .ok_or("partition custom metadata must be a map")?;
                        for (index, value) in rows {
                            let index = global_index(index)?;
                            if partition.rows.contains_key(&index) {
                                if value.as_map().is_some_and(Vec::is_empty) {
                                    partition.custom_meta.remove(&index);
                                } else {
                                    partition
                                        .custom_meta
                                        .insert(index, RawMsgpack::encode(value)?);
                                }
                            }
                        }
                    }
                }
                Ok((
                    "SET_CUSTOM_META_RESPONSE",
                    Body::Fields(vec![(
                        "message".into(),
                        "Successfully set custom_meta".into(),
                    )]),
                ))
            }
            "MERGE_CUSTOM_META" => {
                let partitions = field(body, "partition_custom_meta")
                    .and_then(Value::as_map)
                    .ok_or("partition_custom_meta must be a map")?;
                for (name, meta) in partitions {
                    let name = name.as_str().ok_or("partition name must be a string")?;
                    if let Some(partition) = self.partitions.get_mut(name) {
                        let rows = meta
                            .as_map()
                            .ok_or("partition custom metadata must be a map")?;
                        for (index, value) in rows {
                            let index = global_index(index)?;
                            if !partition.rows.contains_key(&index) {
                                continue;
                            }
                            let incoming = value
                                .as_map()
                                .ok_or("merged custom metadata must be a map")?;
                            if incoming.is_empty() {
                                continue;
                            }
                            // Decode and merge here, in the request loop, so a concurrent update
                            // of another key on this row cannot be lost to a stale client copy.
                            let mut merged = match partition.custom_meta.get(&index) {
                                Some(existing) => existing.decode()?,
                                None => Value::Map(Vec::new()),
                            };
                            let Value::Map(map) = &mut merged else {
                                return Err("stored custom metadata is not a map".into());
                            };
                            for (key, value) in incoming {
                                map.retain(|entry| &entry.0 != key);
                                map.push((key.clone(), value.clone()));
                            }
                            partition
                                .custom_meta
                                .insert(index, RawMsgpack::encode(&merged)?);
                        }
                    }
                }
                Ok((
                    "MERGE_CUSTOM_META_RESPONSE",
                    Body::Fields(vec![(
                        "message".into(),
                        "Successfully merged custom_meta".into(),
                    )]),
                ))
            }
            "MARK_CLEARING" => {
                self.mark_clearing(body)?;
                Ok((
                    "MARK_CLEARING_RESPONSE",
                    Body::Fields(vec![("message".into(), "Mark clearing completed".into())]),
                ))
            }
            "CLEAR_META" => {
                self.clear_meta(body)?;
                Ok((
                    "CLEAR_META_RESPONSE",
                    Body::Fields(vec![(
                        "message".into(),
                        "Clear samples operation completed".into(),
                    )]),
                ))
            }
            "CLEAR_PARTITION" => {
                self.clear_partition(text(body, "partition_id")?);
                Ok((
                    "CLEAR_PARTITION_RESPONSE",
                    Body::Fields(vec![(
                        "message".into(),
                        "Clear partition operation completed".into(),
                    )]),
                ))
            }
            "RESET_CONSUMPTION" => {
                let name = text(body, "partition_id")?;
                if let Some(partition) = self.partitions.get_mut(name) {
                    if let Some(task) = field(body, "task_name").filter(|value| !value.is_nil()) {
                        let task = task.as_str().ok_or("task_name must be a string")?;
                        if let Some(consumed) = partition.consumed.get_mut(task) {
                            consumed.0.clear();
                        }
                        partition.consumed_cursor.insert(task.to_owned(), 0);
                    } else {
                        for consumed in partition.consumed.values_mut() {
                            consumed.0.clear();
                        }
                        partition.consumed_cursor.clear();
                    }
                }
                Ok((
                    "RESET_CONSUMPTION_RESPONSE",
                    Body::Fields(vec![
                        ("partition_id".into(), name.into()),
                        ("success".into(), true.into()),
                        (
                            "message".into(),
                            format!("Consumption reset for partition {name}").into(),
                        ),
                    ]),
                ))
            }
            "KV_RETRIEVE_META" => {
                let name = text(body, "partition_id")?;
                let keys = strings(body, "keys")?;
                let metadata = self.kv_retrieve_meta(name, &keys, flag(body, "create", false)?)?;
                Ok((
                    "KV_RETRIEVE_META_RESPONSE",
                    Body::Metadata(metadata, Vec::new()),
                ))
            }
            "KV_RETRIEVE_KEYS" => {
                let name = text(body, "partition_id")?;
                let indexes = list(body, "global_indexes")?;
                let mut keys = Vec::new();
                if let Some(partition) = self.partitions.get(name) {
                    for index in indexes {
                        keys.push(
                            partition
                                .index_keys
                                .get(&global_index(index)?)
                                .map_or(Value::Nil, |key| key.as_str().into()),
                        );
                    }
                }
                Ok((
                    "KV_RETRIEVE_KEYS_RESPONSE",
                    Body::Fields(vec![("keys".into(), Value::Array(keys))]),
                ))
            }
            "KV_LIST" => Ok(("KV_LIST_RESPONSE", Body::Fields(self.kv_list(body)?))),
            "KV_EXISTS" => Ok(("KV_EXISTS_RESPONSE", Body::Fields(self.kv_exists(body)?))),
            "KV_DESCRIBE" => Ok((
                "KV_DESCRIBE_RESPONSE",
                Body::Fields(self.kv_describe(body)?),
            )),
            "KV_PREPARE_UPDATE" => {
                let (metadata, merge_plan) = self.kv_prepare_update(body)?;
                Ok((
                    "KV_PREPARE_UPDATE_RESPONSE",
                    Body::Metadata(
                        metadata,
                        vec![("merge_plan".into(), Value::Map(merge_plan))],
                    ),
                ))
            }
            "KV_CLEAR_FIELDS" => Ok((
                "KV_CLEAR_FIELDS_RESPONSE",
                Body::Metadata(self.kv_clear_fields(body)?, Vec::new()),
            )),
            "VALIDATE_DUMP_SCHEMA" => {
                self.validate_dump_schema(body)?;
                Ok((
                    "VALIDATE_DUMP_SCHEMA_RESPONSE",
                    Body::Fields(vec![("success".into(), true.into())]),
                ))
            }
            "KV_LINK" => {
                self.kv_link(body)?;
                Ok(("KV_LINK_RESPONSE", Body::Fields(Vec::new())))
            }
            "LINK" => Ok((
                "LINK_RESPONSE",
                Body::Metadata(self.link(body)?, Vec::new()),
            )),
            "CHECK_PUBLIC_WRITE" => {
                let indexes = list(body, "global_indexes")?
                    .iter()
                    .map(global_index)
                    .collect::<StateResult<Vec<_>>>()?;
                self.check_public_not_linked(&indexes, None)?;
                Ok(("CHECK_PUBLIC_WRITE_RESPONSE", Body::Fields(Vec::new())))
            }
            "MARK_UNLINKED_PUBLIC_CLEARING" => {
                let (metadata, keys) = self.mark_unlinked_public_clearing(body)?;
                Ok((
                    "MARK_UNLINKED_PUBLIC_CLEARING_RESPONSE",
                    Body::Metadata(metadata, vec![("keys".into(), Value::Array(keys))]),
                ))
            }
            "SAVE_CONTROLLER_CHECKPOINT" => {
                self.save_checkpoint(text(body, "path")?)?;
                Ok((
                    "SAVE_CONTROLLER_CHECKPOINT_RESPONSE",
                    Body::Fields(vec![("success".into(), true.into())]),
                ))
            }
            "LOAD_CONTROLLER_CHECKPOINT" => {
                self.load_checkpoint(text(body, "path")?)?;
                Ok((
                    "LOAD_CONTROLLER_CHECKPOINT_RESPONSE",
                    Body::Fields(vec![("success".into(), true.into())]),
                ))
            }
            kind => Err(format!("no handler for request type {kind}")),
        }
    }

    fn update_production_status(&mut self, body: &Map) -> StateResult<bool> {
        let name = text(body, "partition_id")?;
        let Some(partition) = self.partitions.get_mut(name) else {
            return Ok(false);
        };
        let indexes = match field(body, "global_indexes") {
            Some(indexes) => indexes
                .as_array()
                .ok_or("global_indexes must be an array")?
                .iter()
                .map(global_index)
                .collect::<StateResult<Vec<_>>>()?,
            None => Vec::new(),
        };
        let fields = match field(body, "field_schema") {
            Some(fields) => fields
                .as_map()
                .ok_or("field_schema must be a map")?
                .as_slice(),
            None => &[],
        };
        let ready = flag(body, "ready", true)?;
        partition.update_production_status(
            &indexes,
            fields,
            field(body, "custom_backend_meta"),
            ready,
        )?;
        if name == PUBLIC_PARTITION && !indexes.is_empty() {
            let names: Vec<_> = fields
                .iter()
                .filter_map(|(name, _)| name.as_str())
                .collect();
            self.refresh_public_links(&indexes, &names, ready);
        }
        Ok(true)
    }

    fn mark_clearing(&mut self, body: &Map) -> StateResult<()> {
        self.mark_clearing_grouped(partition_indexes(body)?)
    }

    pub(crate) fn mark_clearing_rows(
        &mut self,
        indexes: &[GlobalIndex],
        partitions: &[String],
    ) -> StateResult<()> {
        self.mark_clearing_grouped(group_partition_indexes(indexes, partitions)?)
    }

    fn mark_clearing_grouped(
        &mut self,
        partitions: BTreeMap<&str, Vec<GlobalIndex>>,
    ) -> StateResult<()> {
        if let Some(indexes) = partitions.get(PUBLIC_PARTITION) {
            self.check_public_not_linked(indexes, None)?;
        }
        for (name, indexes) in partitions {
            if let Some(partition) = self.partitions.get_mut(name) {
                partition.mark_clearing(&indexes);
            }
        }
        Ok(())
    }

    fn clear_meta(&mut self, body: &Map) -> StateResult<()> {
        self.clear_meta_grouped(partition_indexes(body)?);
        Ok(())
    }

    pub(crate) fn clear_meta_rows(
        &mut self,
        indexes: &[GlobalIndex],
        partitions: &[String],
    ) -> StateResult<()> {
        self.clear_meta_grouped(group_partition_indexes(indexes, partitions)?);
        Ok(())
    }

    fn clear_meta_grouped(&mut self, partitions: BTreeMap<&str, Vec<GlobalIndex>>) {
        for (name, indexes) in partitions {
            if name == PUBLIC_PARTITION {
                self.unlink_public_rows(&indexes);
            }
            if let Some(partition) = self.partitions.get_mut(name) {
                partition.clear_data(&indexes, &mut self.allocator);
            }
        }
    }

    pub(crate) fn clear_partition(&mut self, name: &str) {
        let Some(partition) = self.partitions.shift_remove(name) else {
            return;
        };
        let mut indexes = partition.indexes;
        if name == PUBLIC_PARTITION {
            self.unlink_public_rows(&indexes);
        }
        indexes.sort_unstable();
        for index in indexes {
            self.allocator.release(index);
        }
    }

    fn get_production_status(&self, body: &Map) -> StateResult<Map> {
        let name = text(body, "partition_id")?;
        let requested = list(body, "data_fields")?;
        let missing = vec![
            ("partition_id".into(), name.into()),
            ("global_index".into(), Value::Nil),
            ("production_status".into(), Value::Nil),
        ];
        let Some(partition) = self.partitions.get(name) else {
            return Ok(missing);
        };
        if requested.is_empty()
            || requested.iter().any(|name| {
                !name
                    .as_str()
                    .is_some_and(|name| partition.fields.contains_key(name))
            })
        {
            return Ok(missing);
        }
        // The Python controller returns fields in registration order.
        let fields: Vec<_> = partition
            .fields
            .keys()
            .filter(|name| {
                requested
                    .iter()
                    .any(|requested| requested.as_str() == Some(name))
            })
            .map(|name| &partition.fields[name].ready)
            .collect();
        let rows = partition.sorted_rows();
        let mut status = Vec::with_capacity(rows.len() * fields.len());
        for &row in &rows {
            for ready in &fields {
                status.push(u8::from(ready.get(row)));
            }
        }
        Ok(vec![
            ("partition_id".into(), name.into()),
            (
                "global_index".into(),
                Value::Array(
                    rows.iter()
                        .map(|&row| partition.indexes[row].into())
                        .collect(),
                ),
            ),
            ("production_status".into(), Value::Binary(status)),
        ])
    }

    fn get_consumption_status(&mut self, body: &Map) -> StateResult<Map> {
        let name = text(body, "partition_id")?;
        let task = text(body, "task_name")?;
        let Some(partition) = self.partitions.get_mut(name) else {
            return Ok(vec![
                ("partition_id".into(), name.into()),
                ("global_index".into(), Value::Nil),
                ("consumption_status".into(), Value::Nil),
            ]);
        };
        let rows = partition.sorted_rows();
        let consumed = partition.consumed.entry(task.to_owned()).or_default();
        let status: Vec<u8> = rows
            .iter()
            .map(|&row| u8::from(consumed.get(row)))
            .collect();
        let status = match field(body, "sample_filter").and_then(Value::as_array) {
            Some(filter) if !filter.is_empty() => filter
                .iter()
                .map(|index| {
                    let index = index
                        .as_i64()
                        .ok_or("sample_filter must contain integers")?;
                    let index = if index < 0 {
                        index + status.len() as i64
                    } else {
                        index
                    };
                    usize::try_from(index)
                        .ok()
                        .and_then(|pos| status.get(pos).copied())
                        .ok_or("sample_filter index out of range")
                })
                .collect::<Result<_, _>>()?,
            _ => status,
        };
        Ok(vec![
            ("partition_id".into(), name.into()),
            (
                "global_index".into(),
                Value::Array(
                    rows.iter()
                        .map(|&row| partition.indexes[row].into())
                        .collect(),
                ),
            ),
            ("consumption_status".into(), Value::Binary(status)),
        ])
    }

    fn create_partition(&mut self, name: &str) {
        if self.partitions.contains_key(name) {
            return;
        }
        let mut partition = PartitionState::new(self.allocator.reusable.len());
        for _ in 0..self.preallocate {
            let row = partition.add_row(&mut self.allocator);
            partition.reserved.insert(partition.indexes[row]);
        }
        // Keep pre-allocated `indexes` aligned with `reserved.pop_first()` so sequential
        // activation preserves `LocalRow` order without degrading `seq_sorted`.
        if partition.indexes.len() > 1 {
            partition.indexes.sort_unstable();
            for (row, &index) in partition.indexes.iter().enumerate() {
                partition.rows.insert(index, row);
            }
        }
        self.partitions.insert(name.to_owned(), partition);
    }

    fn get_meta(&mut self, body: &Map) -> StateResult<BatchMeta> {
        let name = text(body, "partition_id")?;
        let mode = field(body, "mode")
            .map_or(Ok("fetch"), |value| value.as_str().ok_or("invalid mode"))?;
        let fields = strings(body, "data_fields")?;
        if mode == "insert" {
            self.create_partition(name);
        }
        if mode == "fetch" && !self.partitions.contains_key(name) {
            return Ok(BatchMeta::default());
        }
        let partition = self
            .partitions
            .get_mut(name)
            .ok_or_else(|| format!("Partition {name} not found"))?;
        let batch_size = || {
            field(body, "batch_size")
                .and_then(Value::as_u64)
                .and_then(|count| usize::try_from(count).ok())
                .ok_or_else(|| "batch_size must be a nonnegative integer".to_owned())
        };
        let rows = match mode {
            "insert" => partition.insert(batch_size()?, &mut self.allocator),
            "fetch" => partition.fetch(&fields, text(body, "task_name")?, batch_size()?),
            "force_fetch" => partition.sorted_rows(),
            _ => return Err(format!("Invalid mode: {mode}")),
        };
        if mode == "fetch" && rows.is_empty() {
            return Ok(BatchMeta::default());
        }
        Ok(partition.snapshot(name, &rows, &fields, mode))
    }
}

fn group_partition_indexes<'a>(
    indexes: &[GlobalIndex],
    names: &'a [String],
) -> StateResult<BTreeMap<&'a str, Vec<GlobalIndex>>> {
    if names.len() != indexes.len() {
        return Err("partition_ids and global_indexes must have the same length".into());
    }
    let mut partitions = BTreeMap::<_, Vec<_>>::new();
    for (name, &index) in names.iter().zip(indexes) {
        partitions.entry(name.as_str()).or_default().push(index);
    }
    for indexes in partitions.values_mut() {
        indexes.sort_unstable();
    }
    Ok(partitions)
}

fn partition_indexes(body: &Map) -> StateResult<BTreeMap<&str, Vec<GlobalIndex>>> {
    let names = list(body, "partition_ids")?;
    let indexes = list(body, "global_indexes")?;
    if names.len() != indexes.len() {
        return Err("partition_ids and global_indexes must have the same length".into());
    }
    let mut partitions = BTreeMap::<_, Vec<_>>::new();
    for (name, index) in names.iter().zip(indexes) {
        let name = name.as_str().ok_or("partition name must be a string")?;
        partitions
            .entry(name)
            .or_default()
            .push(global_index(index)?);
    }
    for indexes in partitions.values_mut() {
        indexes.sort_unstable();
    }
    Ok(partitions)
}

fn strings<'a>(body: &'a Map, key: &str) -> StateResult<Vec<&'a str>> {
    list(body, key)?
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| format!("{key} must contain strings"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regular_schema(shape: &[u64]) -> Value {
        Value::Map(vec![
            ("dtype".into(), "torch.float32".into()),
            (
                "shape".into(),
                Value::Array(shape.iter().copied().map(Value::from).collect()),
            ),
            ("is_nested".into(), false.into()),
            ("is_non_tensor".into(), false.into()),
        ])
    }

    #[test]
    fn bitmap_word_boundaries_and_swap_remove_truncate() {
        let mut bitmap = Bitmap::default();
        for row in [0, 63, 64, 127, 128] {
            bitmap.set(row, true);
        }
        assert_eq!(bitmap.0.len(), 3);
        assert_eq!(bitmap.highest_set(), Some(128));
        assert!(bitmap.get(63) && bitmap.get(64) && bitmap.get(128));
        assert!(!bitmap.get(65) && !bitmap.get(200));

        // Swap-removing row 0 with last row 128 moves the bit into word 0 and drops word 2.
        bitmap.set(0, false);
        bitmap.swap_remove(0, 128);
        assert!(bitmap.get(0));
        assert!(!bitmap.get(128));
        assert_eq!(bitmap.0.len(), 2);

        // Removing the final row in word 1 (row 64 when last == 64) shrinks back to 1 word, then 0.
        bitmap.set(127, false);
        bitmap.swap_remove(64, 64);
        assert_eq!(bitmap.0.len(), 1);
        bitmap.swap_remove(0, 0);
        assert!(bitmap.0.is_empty());
        assert_eq!(bitmap.highest_set(), None);
    }

    #[test]
    fn index_allocator_balances_storage_units_and_reuses_fifo() {
        let mut allocator = IndexAllocator {
            next: 0,
            reusable: vec![VecDeque::new(); 3],
        };
        // Unit 0 already holds two samples from another partition; new allocations steer to units 1 and 2.
        let mut loads = vec![2, 0, 0];
        let first = allocator.allocate(&mut loads);
        let second = allocator.allocate(&mut loads);
        assert_eq!(first % 3, 1);
        assert_eq!(second % 3, 2);
        assert_eq!(loads, vec![2, 1, 1]);
        // Index 0 was skipped and queued for unit 0 rather than lost.
        assert_eq!(allocator.reusable[0].front().copied(), Some(0));

        allocator.release(first);
        loads[1] -= 1;
        let reused = allocator.allocate(&mut loads);
        assert_eq!(reused, first);
    }

    #[test]
    fn partition_sparse_global_indexes_insertion_fifo_and_recycled_slot_fairness() {
        let mut allocator = IndexAllocator {
            next: 1_000_000,
            reusable: vec![VecDeque::new(); 2],
        };
        let mut partition = PartitionState::new(2);

        // Allocate 70 rows (spanning two 64-bit bitmap words) starting at a large sparse GlobalIndex.
        let rows = partition.insert(70, &mut allocator);
        assert_eq!(rows.len(), 70);
        assert!(partition.seq_sorted);
        assert_eq!(partition.active.0.len(), 2);
        let allocated = partition.indexes.clone();

        // Publish only the first 3 rows; requesting batch_size=4 must leave consumption untouched.
        let schema = [(Value::from("x"), regular_schema(&[8]))];
        partition
            .update_production_status(&allocated[..3], &schema, None, true)
            .unwrap();
        assert!(partition.fetch(&["x"], "trainer", 4).is_empty());

        // Publish all 70 rows and consume the first 2 in insertion order.
        partition
            .update_production_status(&allocated[3..], &schema, None, true)
            .unwrap();
        let fetched = partition.fetch(&["x"], "trainer", 2);
        let fetched_indexes: Vec<_> = fetched.iter().map(|&r| partition.indexes[r]).collect();
        assert_eq!(fetched_indexes, allocated[..2]);

        // Clear the first 6 rows (forcing swap_remove from word 1 into word 0).
        partition.clear_data(&allocated[..6], &mut allocator);
        assert_eq!(partition.indexes.len(), 64);
        assert!(!partition.seq_sorted);
        assert_eq!(partition.loads.iter().sum::<usize>(), 64);

        // Insert 4 brand-new rows that reuse the recycled low `GlobalIndex` slots (`allocated[..4]`).
        // Built-in insertion FIFO must serve all 64 older unconsumed rows BEFORE the recycled newcomers.
        let new_rows = partition.insert(4, &mut allocator);
        let new_indexes: Vec<_> = new_rows.iter().map(|&r| partition.indexes[r]).collect();
        assert_eq!(new_indexes, allocated[..4]);
        partition
            .update_production_status(&new_indexes, &schema, None, true)
            .unwrap();

        let older_batch = partition.fetch(&["x"], "trainer", 64);
        let older_indexes: Vec<_> = older_batch.iter().map(|&r| partition.indexes[r]).collect();
        assert_eq!(older_indexes, allocated[6..]);

        let newer_batch = partition.fetch(&["x"], "trainer", 4);
        let newer_indexes: Vec<_> = newer_batch.iter().map(|&r| partition.indexes[r]).collect();
        assert_eq!(newer_indexes, new_indexes);
    }

    #[test]
    fn edge_case_invariants_and_validation() {
        let mut allocator = IndexAllocator {
            next: 0,
            reusable: vec![VecDeque::new(); 1],
        };
        let mut partition = PartitionState::new(1);
        let rows = partition.insert(3, &mut allocator);
        let indexes = partition.indexes.clone();

        // 1. Canonicalizes Torch dtype aliases ("torch.long" == "int64") and rejects invalid dtypes.
        let long_schema = [(
            Value::from("x"),
            Value::Map(vec![
                ("dtype".into(), "torch.long".into()),
                ("shape".into(), Value::Array(vec![4.into()])),
                ("is_nested".into(), false.into()),
                ("is_non_tensor".into(), false.into()),
            ]),
        )];
        let int64_schema = [(
            Value::from("x"),
            Value::Map(vec![
                ("dtype".into(), "int64".into()),
                ("shape".into(), Value::Array(vec![8.into()])),
                ("is_nested".into(), false.into()),
                ("is_non_tensor".into(), false.into()),
            ]),
        )];
        partition
            .update_production_status(&indexes[..1], &long_schema, None, true)
            .unwrap();
        partition
            .update_production_status(&indexes[1..2], &int64_schema, None, true)
            .unwrap();
        let bad_dtype = [(
            Value::from("x"),
            Value::Map(vec![
                ("dtype".into(), "not_a_dtype".into()),
                ("shape".into(), Value::Array(vec![4.into()])),
            ]),
        )];
        assert!(
            partition
                .update_production_status(&indexes[2..], &bad_dtype, None, true)
                .is_err()
        );

        // 2. force_fetch reports status 0 when any requested field is unregistered.
        let partial = partition.snapshot("p", &rows[..1], &["x", "unregistered"], "force_fetch");
        assert_eq!(partial.status, vec![0]);

        // 3. Clearing row 1 must not collapse nested=true while row 2 holds a non-tensor value.
        let non_tensor = [(
            Value::from("x"),
            Value::Map(vec![
                ("dtype".into(), Value::Nil),
                ("shape".into(), Value::Nil),
                ("is_nested".into(), false.into()),
                ("is_non_tensor".into(), true.into()),
            ]),
        )];
        partition
            .update_production_status(&indexes[2..], &non_tensor, None, true)
            .unwrap();
        partition.clear_data(&indexes[1..2], &mut allocator);
        assert!(partition.fields["x"].schema.as_ref().unwrap().nested);

        // 4. Pre-allocated (reserved) rows accept SET_CUSTOM_META and promote on NOTIFY_DATA_UPDATE
        //    without being re-handed out by a subsequent insert.
        let req = |request_type: &str, body: Map| Request {
            request_type: request_type.into(),
            sender_id: "test".into(),
            body,
        };
        let mut controller = ControllerState::new(1, 3);
        let (_, insert_body) = controller
            .apply(&req(
                "GET_META",
                vec![
                    ("partition_id".into(), "pre".into()),
                    ("data_fields".into(), Value::Array(vec!["x".into()])),
                    ("mode".into(), "insert".into()),
                    ("batch_size".into(), 1.into()),
                ],
            ))
            .unwrap();
        let Body::Metadata(inserted, _) = insert_body else {
            panic!("expected metadata");
        };
        assert_eq!(inserted.indexes, vec![0]);

        let (_, ff_body) = controller
            .apply(&req(
                "GET_META",
                vec![
                    ("partition_id".into(), "pre".into()),
                    ("data_fields".into(), Value::Array(vec!["x".into()])),
                    ("mode".into(), "force_fetch".into()),
                ],
            ))
            .unwrap();
        let Body::Metadata(ff_meta, _) = ff_body else {
            panic!("expected metadata");
        };
        assert_eq!(ff_meta.indexes, vec![0, 1, 2]);

        controller
            .apply(&req(
                "SET_CUSTOM_META",
                vec![(
                    "partition_custom_meta".into(),
                    Value::Map(vec![(
                        "pre".into(),
                        Value::Map(vec![(
                            2.into(),
                            Value::Map(vec![("tag".into(), "reserved".into())]),
                        )]),
                    )]),
                )],
            ))
            .unwrap();
        assert!(controller.partitions["pre"].custom_meta.contains_key(&2));

        // Sending an empty map `{}` clears stored custom metadata back to empty.
        controller
            .apply(&req(
                "SET_CUSTOM_META",
                vec![(
                    "partition_custom_meta".into(),
                    Value::Map(vec![(
                        "pre".into(),
                        Value::Map(vec![(2.into(), Value::Map(vec![]))]),
                    )]),
                )],
            ))
            .unwrap();
        assert!(!controller.partitions["pre"].custom_meta.contains_key(&2));

        // 5. MERGE_CUSTOM_META keeps keys it does not send and overwrites the ones it does.
        let merge_tag = |tag: (Value, Value)| {
            let row = Value::Map(vec![tag]);
            let rows = Value::Map(vec![(2.into(), row)]);
            let partition = Value::Map(vec![("pre".into(), rows)]);
            req("MERGE_CUSTOM_META", vec![("partition_custom_meta".into(), partition)])
        };
        controller.apply(&merge_tag(("rank".into(), "worker_a".into()))).unwrap();
        controller.apply(&merge_tag(("attempt".into(), 1.into()))).unwrap();

        let merged = controller.partitions["pre"].custom_meta[&2].decode().unwrap();
        let mut keys: Vec<&str> = merged
            .as_map()
            .unwrap()
            .iter()
            .filter_map(|(key, _)| key.as_str())
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["attempt", "rank"]);

        controller
            .partitions
            .get_mut("pre")
            .unwrap()
            .update_production_status(&ff_meta.indexes, &int64_schema, None, true)
            .unwrap();
        assert!(controller.partitions["pre"].reserved.is_empty());
        let (_, next_body) = controller
            .apply(&req(
                "GET_META",
                vec![
                    ("partition_id".into(), "pre".into()),
                    ("data_fields".into(), Value::Array(vec!["x".into()])),
                    ("mode".into(), "insert".into()),
                    ("batch_size".into(), 1.into()),
                ],
            ))
            .unwrap();
        let Body::Metadata(next_insert, _) = next_body else {
            panic!("expected metadata");
        };
        assert_eq!(next_insert.indexes, vec![3]);
    }

    #[test]
    fn nested_schema_promotion_collapse_and_dtype_atomicity() {
        let mut allocator = IndexAllocator {
            next: 0,
            reusable: vec![VecDeque::new(); 1],
        };
        let mut partition = PartitionState::new(1);
        let rows = partition.insert(3, &mut allocator);
        let indexes = partition.indexes.clone();

        // Uniform [256] promotes to nested when row 2 writes [128], projects per_sample_shapes
        // in requested row order, and collapses back to uniform [256] once row 2 is cleared.
        partition
            .update_production_status(
                &indexes[..2],
                &[(Value::from("x"), regular_schema(&[256]))],
                None,
                true,
            )
            .unwrap();
        partition
            .update_production_status(
                &indexes[2..],
                &[(Value::from("x"), regular_schema(&[128]))],
                None,
                true,
            )
            .unwrap();

        let jagged = partition.snapshot("p", &[rows[2], rows[0]], &["x"], "force_fetch");
        let (_, x_jagged) = jagged.schema.iter().find(|(n, _)| n == "x").unwrap();
        assert!(x_jagged.nested);
        assert_eq!(x_jagged.shape, None);
        assert_eq!(
            x_jagged.sample_shapes.as_ref().unwrap(),
            &vec![Some(vec![128]), Some(vec![256])]
        );

        partition.clear_data(&indexes[2..], &mut allocator);
        let collapsed = partition.snapshot("p", &[0, 1], &["x"], "force_fetch");
        let (_, x_flat) = collapsed.schema.iter().find(|(n, _)| n == "x").unwrap();
        assert!(!x_flat.nested);
        assert_eq!(x_flat.shape, Some(vec![256]));
        assert!(x_flat.sample_shapes.is_none());

        // Rejected dtype mismatch with ready=false must leave existing readiness and dtype intact.
        let bad_int64 = [(
            Value::from("x"),
            Value::Map(vec![
                ("dtype".into(), "int64".into()),
                ("shape".into(), Value::Array(vec![256.into()])),
            ]),
        )];
        assert!(
            partition
                .update_production_status(&indexes[1..2], &bad_int64, None, false)
                .is_err()
        );
        assert_eq!(partition.fetch(&["x"], "reader", 2).len(), 2);
    }

    #[test]
    fn set_num_storage_units_regroups_reusable_slots_and_partition_loads() {
        let mut state = ControllerState::new(1, 0);
        assert!(state.set_num_storage_units(0).is_err());

        let (_, body) = state
            .execute(&Request {
                request_type: "KV_RETRIEVE_META".into(),
                sender_id: "c".into(),
                body: vec![
                    ("partition_id".into(), "p".into()),
                    (
                        "keys".into(),
                        Value::Array(vec!["k0".into(), "k1".into(), "k2".into()]),
                    ),
                    ("create".into(), true.into()),
                ],
            })
            .unwrap();
        let Body::Metadata(meta, _) = body else {
            panic!("expected metadata response");
        };
        assert_eq!(meta.indexes, vec![0, 1, 2]);
        state.clear_meta_rows(&[1], &["p".into()]).unwrap();

        state.set_num_storage_units(3).unwrap();
        assert_eq!(state.num_storage_units(), 3);
        let (_, body) = state
            .execute(&Request {
                request_type: "KV_RETRIEVE_META".into(),
                sender_id: "c".into(),
                body: vec![
                    ("partition_id".into(), "p".into()),
                    ("keys".into(), Value::Array(vec!["k3".into()])),
                    ("create".into(), true.into()),
                ],
            })
            .unwrap();
        let Body::Metadata(reused, _) = body else {
            panic!("expected metadata response");
        };
        // Unit 1 has load 0 (after clearing index 1) and holds recycled slot 1.
        assert_eq!(reused.indexes, vec![1]);
    }

    #[test]
    fn slot_ownership_clearing_window_and_multi_task_consumption() {
        let req = |request_type: &str, body: Map| Request {
            request_type: request_type.into(),
            sender_id: "test".into(),
            body,
        };
        let mut state = ControllerState::new(1, 0);

        // 1. Stale SET_CUSTOM_META on a cleared slot is ignored, and late NOTIFY_DATA_UPDATE
        //    from the old partition after cross-partition slot reuse is rejected.
        let (_, p1_body) = state
            .execute(&req(
                "GET_META",
                vec![
                    ("partition_id".into(), "p1".into()),
                    ("data_fields".into(), Value::Array(vec!["v".into()])),
                    ("mode".into(), "insert".into()),
                    ("batch_size".into(), 1.into()),
                ],
            ))
            .unwrap();
        let Body::Metadata(p1_meta, _) = p1_body else {
            panic!("expected metadata");
        };
        let slot = p1_meta.indexes[0];
        state
            .execute(&req(
                "SET_CUSTOM_META",
                vec![(
                    "partition_custom_meta".into(),
                    Value::Map(vec![(
                        "p1".into(),
                        Value::Map(vec![(
                            slot.into(),
                            Value::Map(vec![("old".into(), 1.into())]),
                        )]),
                    )]),
                )],
            ))
            .unwrap();
        state.clear_meta_rows(&[slot], &["p1".into()]).unwrap();
        state
            .execute(&req(
                "SET_CUSTOM_META",
                vec![(
                    "partition_custom_meta".into(),
                    Value::Map(vec![(
                        "p1".into(),
                        Value::Map(vec![(
                            slot.into(),
                            Value::Map(vec![("ghost".into(), true.into())]),
                        )]),
                    )]),
                )],
            ))
            .unwrap();

        let p2_meta = state.kv_retrieve_meta("p2", &["k0", "k1"], true).unwrap();
        assert_eq!(p2_meta.indexes[0], slot);
        assert!(p2_meta.custom_meta.is_empty());
        assert!(
            state
                .execute(&req(
                    "NOTIFY_DATA_UPDATE",
                    vec![
                        ("partition_id".into(), "p1".into()),
                        ("global_indexes".into(), Value::Array(vec![slot.into()])),
                        (
                            "field_schema".into(),
                            Value::Map(vec![("v".into(), regular_schema(&[1]))]),
                        ),
                    ],
                ))
                .is_err()
        );

        // 2. Backend metadata partial Nil update, multi-task cursors, selective/global reset,
        //    idempotent mark_clearing, and partition-scoped KV_RETRIEVE_KEYS.
        state
            .execute(&req(
                "NOTIFY_DATA_UPDATE",
                vec![
                    ("partition_id".into(), "p2".into()),
                    (
                        "global_indexes".into(),
                        Value::Array(p2_meta.indexes.iter().copied().map(Value::from).collect()),
                    ),
                    (
                        "field_schema".into(),
                        Value::Map(vec![("v".into(), regular_schema(&[2]))]),
                    ),
                    (
                        "custom_backend_meta".into(),
                        Value::Map(vec![(
                            slot.into(),
                            Value::Map(vec![("v".into(), "pinned".into())]),
                        )]),
                    ),
                ],
            ))
            .unwrap();
        state
            .execute(&req(
                "NOTIFY_DATA_UPDATE",
                vec![
                    ("partition_id".into(), "p2".into()),
                    ("global_indexes".into(), Value::Array(vec![slot.into()])),
                    (
                        "field_schema".into(),
                        Value::Map(vec![("v".into(), regular_schema(&[2]))]),
                    ),
                    (
                        "custom_backend_meta".into(),
                        Value::Map(vec![(
                            slot.into(),
                            Value::Map(vec![("v".into(), Value::Nil)]),
                        )]),
                    ),
                ],
            ))
            .unwrap();

        for task in ["logprob", "reward"] {
            let (_, body) = state
                .execute(&req(
                    "GET_META",
                    vec![
                        ("partition_id".into(), "p2".into()),
                        ("data_fields".into(), Value::Array(vec!["v".into()])),
                        ("task_name".into(), task.into()),
                        ("batch_size".into(), 2.into()),
                    ],
                ))
                .unwrap();
            let Body::Metadata(batch, _) = body else {
                panic!("expected metadata");
            };
            assert_eq!(batch.indexes.len(), 2);
            assert_eq!(batch.backend_meta[0].len(), 1);
            assert_eq!(batch.backend_meta[0][0].0, "v");
            assert_eq!(batch.backend_meta[0][0].1.decode().unwrap(), Value::Nil);
        }

        state
            .execute(&req(
                "RESET_CONSUMPTION",
                vec![
                    ("partition_id".into(), "p2".into()),
                    ("task_name".into(), "logprob".into()),
                ],
            ))
            .unwrap();
        assert_eq!(state.partitions["p2"].fetch(&["v"], "logprob", 2).len(), 2);
        assert!(state.partitions["p2"].fetch(&["v"], "reward", 1).is_empty());

        state
            .execute(&req(
                "RESET_CONSUMPTION",
                vec![("partition_id".into(), "p2".into())],
            ))
            .unwrap();
        state
            .mark_clearing_rows(
                &[0, slot, 999],
                &["missing".into(), "p2".into(), "p2".into()],
            )
            .unwrap();
        let remaining = state.partitions["p2"].fetch(&["v"], "reward", 1);
        assert_eq!(remaining.len(), 1);
        assert!(state.partitions["p2"].fetch(&["v"], "reward", 1).is_empty());

        let (_, keys_body) = state
            .execute(&req(
                "KV_RETRIEVE_KEYS",
                vec![
                    ("partition_id".into(), "p2".into()),
                    (
                        "global_indexes".into(),
                        Value::Array(vec![slot.into(), 999.into()]),
                    ),
                ],
            ))
            .unwrap();
        let Body::Fields(fields) = keys_body else {
            panic!("expected fields");
        };
        assert_eq!(
            field(&fields, "keys"),
            Some(&Value::Array(vec!["k0".into(), Value::Nil]))
        );
    }
}
