// Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::Duration;

use rmpv::Value;

use super::ControllerState;
use crate::wire::{Body, Map, Request, field};

pub(super) struct Metrics {
    buckets: Vec<f64>,
    requests: HashMap<String, RequestStats>,
}

struct RequestStats {
    count: u64,
    errors: u64,
    samples: u64,
    seconds: f64,
    counts: Vec<u64>,
}

impl Metrics {
    pub(super) fn new(buckets: Vec<f64>) -> Self {
        Self {
            buckets,
            requests: HashMap::new(),
        }
    }

    pub(super) fn record(&mut self, request: &Request, reply: Option<&Body>, elapsed: Duration) {
        let stats = self
            .requests
            .entry(request.request_type.clone())
            .or_insert_with(|| RequestStats {
                count: 0,
                errors: 0,
                samples: 0,
                seconds: 0.0,
                counts: vec![0; self.buckets.len() + 1],
            });
        stats.count += 1;
        stats.errors += u64::from(reply.is_none());
        let elapsed = elapsed.as_secs_f64();
        stats.seconds += elapsed;
        let bucket = self
            .buckets
            .partition_point(|&threshold| threshold < elapsed);
        stats.counts[bucket] += 1;
        let record_samples = match request.request_type.as_str() {
            "CLEAR_META" => reply.is_some(),
            "NOTIFY_DATA_UPDATE" => {
                matches!(reply, Some(Body::Fields(fields)) if field(fields, "success").and_then(Value::as_bool) == Some(true))
            }
            _ => false,
        };
        if record_samples
            && let Some(indexes) = field(&request.body, "global_indexes").and_then(Value::as_array)
        {
            stats.samples += indexes.len() as u64;
        }
    }
}

impl ControllerState {
    pub(crate) fn metrics_snapshot(&self) -> Map {
        let mut partitions = Vec::new();
        let mut allocated = 0usize;
        for (name, partition) in &self.partitions {
            allocated += partition.indexes.len();
            let active: u32 = partition
                .active
                .0
                .iter()
                .map(|word| word.count_ones())
                .sum();
            let produced: u32 = (0..partition.active.0.len())
                .map(|word| {
                    let active_bits = partition.active.word(word);
                    (active_bits
                        & partition
                            .fields
                            .values()
                            .fold(0, |bits, field| bits | field.ready.word(word)))
                    .count_ones()
                })
                .sum();
            let production_ratio = if active == 0 {
                0.0
            } else {
                f64::from(produced) / f64::from(active)
            };
            let mut production = Vec::new();
            let mut consumption = Vec::new();
            for (name, consumed) in &partition.consumed {
                let count: u32 = (0..partition.active.0.len())
                    .map(|word| (partition.active.word(word) & consumed.word(word)).count_ones())
                    .sum();
                let ratio = if active == 0 {
                    0.0
                } else {
                    f64::from(count) / f64::from(active)
                };
                consumption.push((
                    name.as_str().into(),
                    Value::Map(vec![("consumption_progress".into(), ratio.into())]),
                ));
            }
            for task in std::iter::once("all").chain(partition.consumed.keys().map(String::as_str))
            {
                production.push((
                    task.into(),
                    Value::Map(vec![(
                        "production_progress".into(),
                        production_ratio.into(),
                    )]),
                ));
            }
            partitions.push((
                name.as_str().into(),
                Value::Map(vec![
                    ("total_samples_num".into(), active.into()),
                    ("production_statistics".into(), Value::Map(production)),
                    ("consumption_statistics".into(), Value::Map(consumption)),
                ]),
            ));
        }

        let mut requests = Vec::new();
        if let Some(metrics) = &self.metrics {
            for (kind, stats) in &metrics.requests {
                requests.push((
                    kind.as_str().into(),
                    Value::Map(vec![
                        ("count".into(), stats.count.into()),
                        ("errors".into(), stats.errors.into()),
                        ("samples".into(), stats.samples.into()),
                        ("seconds".into(), stats.seconds.into()),
                        (
                            "buckets".into(),
                            Value::Array(stats.counts.iter().copied().map(Value::from).collect()),
                        ),
                    ]),
                ));
            }
        }
        let reusable: usize = self.allocator.reusable.iter().map(|unit| unit.len()).sum();
        vec![
            ("partitions".into(), Value::Map(partitions)),
            ("global_index_allocated".into(), allocated.into()),
            ("global_index_reusable".into(), reusable.into()),
            ("requests".into(), Value::Map(requests)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(request_type: &str, body: Map) -> Request {
        Request {
            request_type: request_type.into(),
            sender_id: "test".into(),
            body,
        }
    }

    #[test]
    fn metrics_record_buckets_samples_errors_and_partition_progress() {
        let mut metrics = Metrics::new(vec![0.01, 0.1]);
        let notify_req = req(
            "NOTIFY_DATA_UPDATE",
            vec![(
                "global_indexes".into(),
                Value::Array(vec![0.into(), 1.into()]),
            )],
        );
        // Accepted notify lands in bucket 0 (<= 10 ms) and counts 2 samples.
        metrics.record(
            &notify_req,
            Some(&Body::Fields(vec![("success".into(), true.into())])),
            Duration::from_millis(5),
        );
        // Dropped notify (success=false) lands in bucket 1 (<= 100 ms) and adds 0 samples, 0 errors.
        metrics.record(
            &notify_req,
            Some(&Body::Fields(vec![("success".into(), false.into())])),
            Duration::from_millis(50),
        );
        // Failed notify (reply=None) lands in +Inf overflow bucket 2 and increments errors without samples.
        metrics.record(&notify_req, None, Duration::from_millis(250));

        let stats = &metrics.requests["NOTIFY_DATA_UPDATE"];
        assert_eq!(stats.count, 3);
        assert_eq!(stats.errors, 1);
        assert_eq!(stats.samples, 2);
        assert_eq!(stats.counts, vec![1, 1, 1]);

        let mut state = ControllerState::new(1, 0);
        state.enable_metrics(vec![1.0]);
        let meta = state.kv_retrieve_meta("p_m", &["k0", "k1"], true).unwrap();
        let [k0_idx, k1_idx] = meta.indexes[..] else {
            panic!("expected two indexes");
        };
        state
            .execute(&req(
                "NOTIFY_DATA_UPDATE",
                vec![
                    ("partition_id".into(), "p_m".into()),
                    ("global_indexes".into(), Value::Array(vec![k0_idx.into()])),
                    (
                        "field_schema".into(),
                        Value::Map(vec![(
                            "x".into(),
                            Value::Map(vec![
                                ("dtype".into(), "int64".into()),
                                ("shape".into(), Value::Array(vec![1.into()])),
                            ]),
                        )]),
                    ),
                ],
            ))
            .unwrap();
        state
            .execute(&req(
                "GET_META",
                vec![
                    ("partition_id".into(), "p_m".into()),
                    ("data_fields".into(), Value::Array(vec!["x".into()])),
                    ("mode".into(), "fetch".into()),
                    ("task_name".into(), "trainer".into()),
                    ("batch_size".into(), 1.into()),
                ],
            ))
            .unwrap();
        assert!(state.execute(&req("UNKNOWN_RPC", vec![])).is_err());

        let snapshot = state.metrics_snapshot();
        assert_eq!(
            field(&snapshot, "global_index_allocated").unwrap().as_u64(),
            Some(2)
        );
        let p_m = field(
            field(&snapshot, "partitions").unwrap().as_map().unwrap(),
            "p_m",
        )
        .unwrap()
        .as_map()
        .unwrap();
        assert_eq!(field(p_m, "total_samples_num").unwrap().as_u64(), Some(2));
        let prod = field(p_m, "production_statistics")
            .unwrap()
            .as_map()
            .unwrap();
        let all_prod = field(prod, "all").unwrap().as_map().unwrap();
        assert_eq!(
            field(all_prod, "production_progress").unwrap().as_f64(),
            Some(0.5)
        );
        let cons = field(p_m, "consumption_statistics")
            .unwrap()
            .as_map()
            .unwrap();
        let trainer_cons = field(cons, "trainer").unwrap().as_map().unwrap();
        assert_eq!(
            field(trainer_cons, "consumption_progress")
                .unwrap()
                .as_f64(),
            Some(0.5)
        );

        state
            .execute(&req(
                "CLEAR_META",
                vec![
                    ("global_indexes".into(), Value::Array(vec![k1_idx.into()])),
                    ("partition_ids".into(), Value::Array(vec!["p_m".into()])),
                ],
            ))
            .unwrap();
        let after_clear = state.metrics_snapshot();
        assert_eq!(
            field(&after_clear, "global_index_reusable")
                .unwrap()
                .as_u64(),
            Some(1)
        );
        let reqs = field(&after_clear, "requests").unwrap().as_map().unwrap();
        let clear_stats = field(reqs, "CLEAR_META").unwrap().as_map().unwrap();
        assert_eq!(field(clear_stats, "samples").unwrap().as_u64(), Some(1));
        let unknown_stats = field(reqs, "UNKNOWN_RPC").unwrap().as_map().unwrap();
        assert_eq!(field(unknown_stats, "errors").unwrap().as_u64(), Some(1));
    }
}
