// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::time::{SystemTime, UNIX_EPOCH};

use rmpv::Value;
use uuid::Uuid;

use crate::Result;

pub(crate) type Map = Vec<(Value, Value)>;

pub(crate) struct Request {
    pub(crate) request_type: String,
    pub(crate) sender_id: String,
    pub(crate) body: Map,
}

pub(crate) fn field<'a>(fields: &'a Map, key: &str) -> Option<&'a Value> {
    fields
        .iter()
        .find(|(name, _)| name.as_str() == Some(key))
        .map(|(_, value)| value)
}

pub(crate) fn text<'a>(fields: &'a Map, key: &str) -> std::result::Result<&'a str, String> {
    field(fields, key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or invalid {key}"))
}

pub(crate) fn list<'a>(fields: &'a Map, key: &str) -> std::result::Result<&'a [Value], String> {
    field(fields, key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("missing or invalid {key}"))
}

pub(crate) fn decode(frames: &[Vec<u8>]) -> std::result::Result<Request, String> {
    let [frame] = frames else {
        return Err(format!(
            "expected a single MessagePack frame, received {} frames",
            frames.len()
        ));
    };
    let mut input = frame.as_slice();
    let value = rmpv::decode::read_value(&mut input).map_err(|error| error.to_string())?;
    if !input.is_empty() {
        return Err("trailing bytes in MessagePack frame".into());
    }
    let Value::Map(mut fields) = value else {
        return Err("request envelope must be a map".into());
    };
    text(&fields, "request_id")?;
    if !field(&fields, "receiver_id").is_some_and(|value| value.is_nil() || value.is_str())
        || !field(&fields, "timestamp").is_some_and(|value| value.as_f64().is_some())
    {
        return Err("invalid receiver_id or timestamp".into());
    }
    let request_type = text(&fields, "request_type")?.to_owned();
    let sender_id = text(&fields, "sender_id")?.to_owned();
    let body = fields
        .iter()
        .position(|(key, _)| key.as_str() == Some("body"))
        .and_then(|index| match fields.swap_remove(index).1 {
            Value::Map(body) => Some(body),
            _ => None,
        })
        .ok_or("missing or invalid body")?;
    Ok(Request {
        request_type,
        sender_id,
        body,
    })
}

#[derive(Default)]
pub(crate) struct BatchMeta {
    pub(crate) partition_id: String,
    pub(crate) indexes: Vec<u64>,
    pub(crate) schema: Map,
    pub(crate) status: Vec<u8>,
    pub(crate) backend_meta: Vec<Value>,
    pub(crate) custom_meta: Vec<Value>,
    pub(crate) links: Vec<Value>,
}

impl BatchMeta {
    pub(crate) fn encode(self) -> Result<Value> {
        let mut metadata = vec![
            (
                "partition_ids".into(),
                Value::Array(vec![self.partition_id.into(); self.indexes.len()]),
            ),
            (
                "global_indexes".into(),
                Value::Array(self.indexes.into_iter().map(Value::from).collect()),
            ),
            ("field_schema".into(), Value::Map(self.schema)),
            ("production_status".into(), Value::Binary(self.status)),
        ];
        for (name, rows) in [
            ("custom_meta", self.custom_meta),
            ("custom_backend_meta", self.backend_meta),
            ("links", self.links),
        ] {
            if rows
                .iter()
                .any(|row| row.as_map().is_some_and(|map| !map.is_empty()))
            {
                metadata.push((name.into(), Value::Array(rows)));
            }
        }
        let mut payload = Vec::new();
        rmpv::encode::write_value(&mut payload, &Value::Map(metadata))?;
        // The Python client uses extension 6 for native MessagePack BatchMeta.
        Ok(Value::Ext(6, payload))
    }
}

pub(crate) enum Body {
    Fields(Map),
    Metadata(BatchMeta, Map),
}

pub(crate) struct Reply {
    pub(crate) receiver: Option<String>,
    pub(crate) kind: &'static str,
    pub(crate) body: Body,
}

impl Reply {
    pub(crate) fn error(receiver: Option<String>, message: String) -> Self {
        Self {
            receiver,
            kind: "REQUEST_ERROR",
            body: Body::Fields(vec![
                ("success".into(), false.into()),
                ("message".into(), message.into()),
            ]),
        }
    }

    pub(crate) fn encode(self, controller_id: &str) -> Result<Vec<u8>> {
        let body = match self.body {
            Body::Fields(fields) => fields,
            Body::Metadata(batch, mut fields) => {
                fields.push(("metadata".into(), batch.encode()?));
                fields
            }
        };
        let request_id = Uuid::new_v4().simple().to_string();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let message = Value::Map(vec![
            ("request_type".into(), self.kind.into()),
            ("sender_id".into(), controller_id.into()),
            (
                "receiver_id".into(),
                self.receiver.map_or(Value::Nil, Value::from),
            ),
            ("request_id".into(), request_id[..8].into()),
            ("timestamp".into(), timestamp.into()),
            ("body".into(), Value::Map(body)),
        ]);
        let mut frame = Vec::new();
        rmpv::encode::write_value(&mut frame, &message)?;
        Ok(frame)
    }
}

pub(crate) fn handshake(frames: &[Vec<u8>]) -> Reply {
    match decode(frames) {
        Ok(request) if request.request_type == "HANDSHAKE" => Reply {
            receiver: None,
            kind: "HANDSHAKE_ACK",
            body: Body::Fields(Vec::new()),
        },
        Ok(request) => Reply::error(
            Some(request.sender_id),
            format!("no handler for request type {}", request.request_type),
        ),
        Err(error) => Reply::error(None, error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_value(value: &Value) -> Vec<u8> {
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, value).unwrap();
        buf
    }

    fn request_frame(request_type: &str, sender_id: &str, body: Value) -> Vec<u8> {
        encode_value(&Value::Map(vec![
            ("request_id".into(), "req00001".into()),
            ("sender_id".into(), sender_id.into()),
            ("receiver_id".into(), Value::Nil),
            ("request_type".into(), request_type.into()),
            ("timestamp".into(), 1.5_f64.into()),
            ("body".into(), body),
        ]))
    }

    #[test]
    fn decode_and_handshake_validate_envelopes() {
        let valid = request_frame(
            "GET_META",
            "client-1",
            Value::Map(vec![("partition_id".into(), "p0".into())]),
        );
        let decoded = decode(std::slice::from_ref(&valid)).unwrap();
        assert_eq!(decoded.request_type, "GET_META");
        assert_eq!(decoded.sender_id, "client-1");
        assert_eq!(text(&decoded.body, "partition_id").unwrap(), "p0");

        assert!(
            decode(&[])
                .err()
                .unwrap()
                .contains("single MessagePack frame")
        );
        assert!(
            decode(&[valid.clone(), valid.clone()])
                .err()
                .unwrap()
                .contains("single MessagePack frame")
        );

        let mut trailing = valid.clone();
        trailing.push(0xc0);
        assert!(
            decode(&[trailing])
                .err()
                .unwrap()
                .contains("trailing bytes")
        );

        assert!(
            decode(&[encode_value(&Value::Array(vec![1.into()]))])
                .err()
                .unwrap()
                .contains("must be a map")
        );
        assert!(
            decode(&[request_frame("GET_META", "c", Value::Array(vec![]))])
                .err()
                .unwrap()
                .contains("invalid body")
        );

        let bad_ts = encode_value(&Value::Map(vec![
            ("request_id".into(), "req1".into()),
            ("sender_id".into(), "c".into()),
            ("receiver_id".into(), Value::Nil),
            ("request_type".into(), "GET_META".into()),
            ("timestamp".into(), "not-a-float".into()),
            ("body".into(), Value::Map(vec![])),
        ]));
        assert!(decode(&[bad_ts]).err().unwrap().contains("timestamp"));

        let hs_ok = handshake(&[request_frame("HANDSHAKE", "c1", Value::Map(vec![]))]);
        assert_eq!(hs_ok.kind, "HANDSHAKE_ACK");
        assert_eq!(hs_ok.receiver, None);

        let hs_wrong = handshake(&[valid]);
        assert_eq!(hs_wrong.kind, "REQUEST_ERROR");
        assert_eq!(hs_wrong.receiver.as_deref(), Some("client-1"));
    }

    #[test]
    fn batch_meta_ext6_omits_empty_maps_and_encodes_non_empty_rows() {
        let empty_rows = BatchMeta {
            partition_id: "train".into(),
            indexes: vec![7],
            schema: vec![],
            status: vec![1],
            backend_meta: vec![Value::Map(vec![])],
            custom_meta: vec![Value::Map(vec![])],
            links: vec![Value::Map(vec![])],
        };
        let Value::Ext(code, payload) = empty_rows.encode().unwrap() else {
            panic!("expected Ext");
        };
        assert_eq!(code, 6);
        let mut slice = payload.as_slice();
        let Value::Map(fields) = rmpv::decode::read_value(&mut slice).unwrap() else {
            panic!("expected map");
        };
        assert!(field(&fields, "custom_meta").is_none());
        assert!(field(&fields, "custom_backend_meta").is_none());
        assert!(field(&fields, "links").is_none());

        let populated = BatchMeta {
            partition_id: "train".into(),
            indexes: vec![7],
            schema: vec![],
            status: vec![1],
            backend_meta: vec![],
            custom_meta: vec![Value::Map(vec![("step".into(), 3.into())])],
            links: vec![Value::Map(vec![("prompt".into(), 0.into())])],
        };
        let reply = Reply {
            receiver: Some("client-1".into()),
            kind: "GET_META_RESPONSE",
            body: Body::Metadata(populated, vec![]),
        };
        let frame = reply.encode("ctrl-1").unwrap();
        let mut frame_slice = frame.as_slice();
        let Value::Map(envelope) = rmpv::decode::read_value(&mut frame_slice).unwrap() else {
            panic!("expected reply map");
        };
        let body = field(&envelope, "body").unwrap().as_map().unwrap();
        let Some(Value::Ext(6, ext_bytes)) = field(body, "metadata") else {
            panic!("expected metadata Ext(6)");
        };
        let mut ext_slice = ext_bytes.as_slice();
        let Value::Map(meta_map) = rmpv::decode::read_value(&mut ext_slice).unwrap() else {
            panic!("expected metadata map");
        };
        assert_eq!(list(&meta_map, "custom_meta").unwrap().len(), 1);
        assert_eq!(list(&meta_map, "links").unwrap().len(), 1);
        assert!(field(&meta_map, "custom_backend_meta").is_none());
    }
}
