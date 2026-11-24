// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rmp::encode as msgpack;
use rmpv::Value;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Result;

// Matches `CUSTOM_TYPE_BATCH_META = 6` in `transfer_queue/utils/serial_utils.py`.
pub(crate) const BATCH_META_EXT_CODE: i8 = 6;

static NEXT_REPLY_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) type Map = Vec<(Value, Value)>;

/// Pre-encoded native MessagePack value stored as an immutable reference-counted byte slice,
/// avoiding recursive `rmpv::Value` clones under the state mutex and re-serialization on replies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RawMsgpack(Arc<[u8]>);

impl RawMsgpack {
    pub(crate) fn encode(value: &Value) -> std::result::Result<Self, String> {
        let mut bytes = Vec::new();
        rmpv::encode::write_value(&mut bytes, value).map_err(|error| error.to_string())?;
        Ok(Self(bytes.into()))
    }

    pub(crate) fn decode(&self) -> std::result::Result<Value, String> {
        let mut slice = self.0.as_ref();
        let value = rmpv::decode::read_value(&mut slice).map_err(|error| error.to_string())?;
        if !slice.is_empty() {
            return Err("trailing bytes in stored MessagePack metadata".into());
        }
        Ok(value)
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl Serialize for RawMsgpack {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serde_bytes::Bytes::new(&self.0).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RawMsgpack {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let bytes = serde_bytes::ByteBuf::deserialize(deserializer)?;
        Ok(Self(bytes.into_vec().into()))
    }
}

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

pub(crate) struct FieldSchemaWire {
    pub(crate) dtype: Option<String>,
    pub(crate) shape: Option<Vec<u64>>,
    pub(crate) nested: bool,
    pub(crate) non_tensor: bool,
    pub(crate) sample_shapes: Option<Vec<Option<Vec<u64>>>>,
}

#[derive(Default)]
pub(crate) struct BatchMeta {
    pub(crate) partition_id: String,
    pub(crate) indexes: Vec<u64>,
    pub(crate) schema: Vec<(String, FieldSchemaWire)>,
    pub(crate) status: Vec<u8>,
    pub(crate) backend_meta: Vec<Vec<(String, RawMsgpack)>>,
    pub(crate) custom_meta: Vec<Option<RawMsgpack>>,
    pub(crate) links: Vec<Vec<(String, u64)>>,
}

impl BatchMeta {
    /// Stream this `BatchMeta` directly into `buf` as a MessagePack `Ext(6, ...)` frame without
    /// allocating repeated `partition_id` strings or an intermediate payload `Vec<u8>`.
    pub(crate) fn encode_into(&self, buf: &mut Vec<u8>) -> Result<()> {
        let header_pos = buf.len();
        buf.extend_from_slice(&[0xc9, 0, 0, 0, 0, BATCH_META_EXT_CODE as u8]);
        let payload_start = buf.len();

        let has_custom = self.custom_meta.iter().any(Option::is_some);
        let has_backend = self.backend_meta.iter().any(|row| !row.is_empty());
        let has_links = self.links.iter().any(|row| !row.is_empty());
        let map_len = 4 + u32::from(has_custom) + u32::from(has_backend) + u32::from(has_links);
        msgpack::write_map_len(buf, map_len)?;

        let row_count = u32::try_from(self.indexes.len())?;
        msgpack::write_str(buf, "partition_ids")?;
        msgpack::write_array_len(buf, row_count)?;
        for _ in 0..self.indexes.len() {
            msgpack::write_str(buf, &self.partition_id)?;
        }

        msgpack::write_str(buf, "global_indexes")?;
        msgpack::write_array_len(buf, row_count)?;
        for &index in &self.indexes {
            msgpack::write_uint(buf, index)?;
        }

        msgpack::write_str(buf, "field_schema")?;
        msgpack::write_map_len(buf, u32::try_from(self.schema.len())?)?;
        for (name, schema) in &self.schema {
            msgpack::write_str(buf, name)?;
            let schema_len = 4 + u32::from(schema.sample_shapes.is_some());
            msgpack::write_map_len(buf, schema_len)?;

            msgpack::write_str(buf, "dtype")?;
            match &schema.dtype {
                Some(dtype) => msgpack::write_str(buf, dtype)?,
                None => msgpack::write_nil(buf)?,
            }

            msgpack::write_str(buf, "shape")?;
            write_shape(buf, schema.shape.as_deref())?;

            msgpack::write_str(buf, "is_nested")?;
            msgpack::write_bool(buf, schema.nested)?;

            msgpack::write_str(buf, "is_non_tensor")?;
            msgpack::write_bool(buf, schema.non_tensor)?;

            if let Some(shapes) = &schema.sample_shapes {
                msgpack::write_str(buf, "per_sample_shapes")?;
                msgpack::write_array_len(buf, u32::try_from(shapes.len())?)?;
                for shape in shapes {
                    write_shape(buf, shape.as_deref())?;
                }
            }
        }

        msgpack::write_str(buf, "production_status")?;
        msgpack::write_bin(buf, &self.status)?;

        if has_custom {
            msgpack::write_str(buf, "custom_meta")?;
            msgpack::write_array_len(buf, u32::try_from(self.custom_meta.len())?)?;
            for entry in &self.custom_meta {
                match entry {
                    Some(raw) => buf.extend_from_slice(raw.as_bytes()),
                    None => {
                        msgpack::write_map_len(buf, 0)?;
                    }
                }
            }
        }

        if has_backend {
            msgpack::write_str(buf, "custom_backend_meta")?;
            msgpack::write_array_len(buf, u32::try_from(self.backend_meta.len())?)?;
            for row in &self.backend_meta {
                msgpack::write_map_len(buf, u32::try_from(row.len())?)?;
                for (name, raw) in row {
                    msgpack::write_str(buf, name)?;
                    buf.extend_from_slice(raw.as_bytes());
                }
            }
        }

        if has_links {
            msgpack::write_str(buf, "links")?;
            msgpack::write_array_len(buf, u32::try_from(self.links.len())?)?;
            for row in &self.links {
                msgpack::write_map_len(buf, u32::try_from(row.len())?)?;
                for (name, source) in row {
                    msgpack::write_str(buf, name)?;
                    msgpack::write_uint(buf, *source)?;
                }
            }
        }

        let payload_len = u32::try_from(buf.len() - payload_start)?;
        buf[header_pos + 1..header_pos + 5].copy_from_slice(&payload_len.to_be_bytes());
        Ok(())
    }
}

fn write_shape(buf: &mut Vec<u8>, shape: Option<&[u64]>) -> Result<()> {
    match shape {
        Some(dims) => {
            msgpack::write_array_len(buf, u32::try_from(dims.len())?)?;
            for &dim in dims {
                msgpack::write_uint(buf, dim)?;
            }
        }
        None => msgpack::write_nil(buf)?,
    }
    Ok(())
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
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let request_id = next_reply_id();
        let mut frame = Vec::with_capacity(256);

        msgpack::write_map_len(&mut frame, 6)?;
        msgpack::write_str(&mut frame, "request_type")?;
        msgpack::write_str(&mut frame, self.kind)?;

        msgpack::write_str(&mut frame, "sender_id")?;
        msgpack::write_str(&mut frame, controller_id)?;

        msgpack::write_str(&mut frame, "receiver_id")?;
        match &self.receiver {
            Some(receiver) => msgpack::write_str(&mut frame, receiver)?,
            None => msgpack::write_nil(&mut frame)?,
        }

        msgpack::write_str(&mut frame, "request_id")?;
        let id_str = std::str::from_utf8(&request_id).expect("hex digits are ASCII");
        msgpack::write_str(&mut frame, id_str)?;

        msgpack::write_str(&mut frame, "timestamp")?;
        msgpack::write_f64(&mut frame, timestamp)?;

        msgpack::write_str(&mut frame, "body")?;
        match self.body {
            Body::Fields(fields) => {
                msgpack::write_map_len(&mut frame, u32::try_from(fields.len())?)?;
                for (key, value) in fields {
                    rmpv::encode::write_value(&mut frame, &key)?;
                    rmpv::encode::write_value(&mut frame, &value)?;
                }
            }
            Body::Metadata(batch, extra) => {
                msgpack::write_map_len(&mut frame, u32::try_from(extra.len() + 1)?)?;
                msgpack::write_str(&mut frame, "metadata")?;
                batch.encode_into(&mut frame)?;
                for (key, value) in extra {
                    rmpv::encode::write_value(&mut frame, &key)?;
                    rmpv::encode::write_value(&mut frame, &value)?;
                }
            }
        }
        Ok(frame)
    }
}

fn next_reply_id() -> [u8; 8] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let seq = NEXT_REPLY_ID.fetch_add(1, Ordering::Relaxed) as u32;
    let mut out = [0u8; 8];
    for (idx, slot) in out.iter_mut().enumerate() {
        let shift = (7 - idx) * 4;
        *slot = HEX[((seq >> shift) & 0x0f) as usize];
    }
    out
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
        Err(error) => Reply::error(None, format!("undecodable request: {error}")),
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
    fn batch_meta_ext6_omits_empty_maps_and_streams_non_empty_rows() {
        let empty_rows = BatchMeta {
            partition_id: "train".into(),
            indexes: vec![7],
            schema: vec![],
            status: vec![1],
            backend_meta: vec![vec![]],
            custom_meta: vec![None],
            links: vec![vec![]],
        };
        let mut raw_ext = Vec::new();
        empty_rows.encode_into(&mut raw_ext).unwrap();
        let mut ext_reader = raw_ext.as_slice();
        let Value::Ext(code, payload) = rmpv::decode::read_value(&mut ext_reader).unwrap() else {
            panic!("expected Ext");
        };
        assert_eq!(code, BATCH_META_EXT_CODE);
        let mut slice = payload.as_slice();
        let Value::Map(fields) = rmpv::decode::read_value(&mut slice).unwrap() else {
            panic!("expected map");
        };
        assert!(field(&fields, "custom_meta").is_none());
        assert!(field(&fields, "custom_backend_meta").is_none());
        assert!(field(&fields, "links").is_none());

        let populated = BatchMeta {
            partition_id: "train".into(),
            indexes: vec![7, 9],
            schema: vec![(
                "ids".into(),
                FieldSchemaWire {
                    dtype: Some("int64".into()),
                    shape: Some(vec![4]),
                    nested: false,
                    non_tensor: false,
                    sample_shapes: None,
                },
            )],
            status: vec![1, 1],
            backend_meta: vec![],
            custom_meta: vec![
                Some(RawMsgpack::encode(&Value::Map(vec![("step".into(), 3.into())])).unwrap()),
                None,
            ],
            links: vec![vec![("prompt".into(), 0)], vec![]],
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
        assert_eq!(text(&envelope, "request_id").unwrap().len(), 8);
        let body = field(&envelope, "body").unwrap().as_map().unwrap();
        let Some(Value::Ext(6, ext_bytes)) = field(body, "metadata") else {
            panic!("expected metadata Ext(6)");
        };
        let mut ext_slice = ext_bytes.as_slice();
        let Value::Map(meta_map) = rmpv::decode::read_value(&mut ext_slice).unwrap() else {
            panic!("expected metadata map");
        };
        assert_eq!(
            list(&meta_map, "partition_ids").unwrap(),
            &[Value::from("train"), Value::from("train")]
        );
        let custom = list(&meta_map, "custom_meta").unwrap();
        assert_eq!(custom.len(), 2);
        assert_eq!(custom[1], Value::Map(vec![]));
        assert_eq!(list(&meta_map, "links").unwrap().len(), 2);
        assert!(field(&meta_map, "custom_backend_meta").is_none());
    }
}
