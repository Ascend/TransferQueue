// Copyright 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::ffi::c_void;
use std::hint;
use std::io;
use std::net::IpAddr;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::Result;
use crate::engine::lock_state;
use crate::log::{log_debug, log_error, log_info, log_warn};
use crate::state::ControllerState;
use crate::wire::{self, Body, Reply, Request};

const BATCH_DRAIN_LIMIT: usize = 256;
// Spin briefly before sleeping in zmq::poll so back-to-back ping-pong RPCs avoid kernel wakeups.
const IDLE_SPIN_WINDOW: Duration = Duration::from_micros(100);
const DEFAULT_ZMQ_MAX_SOCKETS: i32 = 4096;

struct Routed {
    route: Vec<u8>,
    data: Vec<Vec<u8>>,
}

struct ZmqContext(*mut c_void);

// SAFETY: libzmq contexts (`zmq_ctx_t`) are thread-safe across threads.
unsafe impl Send for ZmqContext {}
unsafe impl Sync for ZmqContext {}

impl ZmqContext {
    fn new_with_socket_budget() -> Result<(Self, usize)> {
        let raw = unsafe { zmq_sys::zmq_ctx_new() };
        if raw.is_null() {
            return Err(zmq::Error::from_raw(unsafe { zmq_sys::zmq_errno() }).into());
        }
        let context = Self(raw);
        let limit = unsafe { zmq_sys::zmq_ctx_get(context.0, zmq_sys::ZMQ_SOCKET_LIMIT as i32) };
        if limit < 0 {
            return Err(zmq::Error::from_raw(unsafe { zmq_sys::zmq_errno() }).into());
        }
        let target = configured_max_sockets()?.min(limit);
        if unsafe { zmq_sys::zmq_ctx_set(context.0, zmq_sys::ZMQ_MAX_SOCKETS as i32, target) } != 0
        {
            return Err(zmq::Error::from_raw(unsafe { zmq_sys::zmq_errno() }).into());
        }
        let actual = unsafe { zmq_sys::zmq_ctx_get(context.0, zmq_sys::ZMQ_MAX_SOCKETS as i32) };
        if actual < 0 {
            return Err(zmq::Error::from_raw(unsafe { zmq_sys::zmq_errno() }).into());
        }
        Ok((context, actual as usize))
    }

    fn router_socket(&self) -> std::result::Result<zmq::Socket, zmq::Error> {
        let sock = unsafe { zmq_sys::zmq_socket(self.0, zmq_sys::ZMQ_ROUTER as i32) };
        if sock.is_null() {
            return Err(zmq::Error::from_raw(unsafe { zmq_sys::zmq_errno() }));
        }
        // SAFETY: `sock` was freshly created on `self.0` and is closed before `_context` drops.
        Ok(unsafe { zmq::Socket::from_raw(sock) })
    }
}

impl Drop for ZmqContext {
    fn drop(&mut self) {
        if !self.0.is_null() {
            loop {
                let rc = unsafe { zmq_sys::zmq_ctx_term(self.0) };
                if rc == 0
                    || zmq::Error::from_raw(unsafe { zmq_sys::zmq_errno() }) != zmq::Error::EINTR
                {
                    break;
                }
            }
        }
    }
}

// Field order is load-bearing: `handshake` and `request` must drop before `_context` terminates.
pub(crate) struct Sockets {
    handshake: zmq::Socket,
    request: zmq::Socket,
    _context: ZmqContext,
    ports: (u16, u16),
    max_sockets: usize,
}

impl Sockets {
    pub(crate) fn bind(ip: IpAddr) -> Result<Self> {
        let (context, max_sockets) = ZmqContext::new_with_socket_budget()?;
        let (handshake, handshake_port) = bind_router(&context, ip)?;
        let (request, request_port) = bind_router(&context, ip)?;
        Ok(Self {
            handshake,
            request,
            _context: context,
            ports: (handshake_port, request_port),
            max_sockets,
        })
    }

    pub(crate) fn ports(&self) -> (u16, u16) {
        self.ports
    }

    #[cfg(test)]
    pub(crate) fn max_sockets(&self) -> usize {
        self.max_sockets
    }

    pub(crate) fn run(
        self,
        stopping: &AtomicBool,
        controller_id: &str,
        state: &Mutex<ControllerState>,
    ) -> Result<()> {
        log_info!(
            controller_id,
            "started ZMQ worker (ports={:?}, max_sockets={})",
            self.ports,
            self.max_sockets
        );
        let mut idle_since: Option<Instant> = None;
        while !stopping.load(Ordering::Acquire) {
            let mut progressed = false;
            for _ in 0..BATCH_DRAIN_LIMIT {
                if stopping.load(Ordering::Acquire) {
                    break;
                }
                let Some(message) = receive(&self.handshake, controller_id)? else {
                    break;
                };
                progressed = true;
                match wire::handshake(&message.data).encode(controller_id) {
                    Ok(reply) => send(&self.handshake, &message.route, &reply, controller_id),
                    Err(error) => log_error!(controller_id, "handshake encode error: {error}"),
                }
            }
            for _ in 0..BATCH_DRAIN_LIMIT {
                if stopping.load(Ordering::Acquire) {
                    break;
                }
                let Some(request) = receive(&self.request, controller_id)? else {
                    break;
                };
                progressed = true;
                let reply = dispatch(controller_id, state, &request.data);
                let receiver = reply.receiver.clone();
                let encoded = reply.encode(controller_id).or_else(|error| {
                    log_error!(controller_id, "reply encode error: {error}");
                    Reply::error(receiver, error.to_string()).encode(controller_id)
                });
                match encoded {
                    Ok(frame) => send(&self.request, &request.route, &frame, controller_id),
                    Err(error) => log_error!(controller_id, "fallback reply encode error: {error}"),
                }
            }
            if progressed {
                idle_since = None;
                continue;
            }
            if idle_since.get_or_insert_with(Instant::now).elapsed() < IDLE_SPIN_WINDOW {
                hint::spin_loop();
                continue;
            }
            let mut items = [
                self.handshake.as_poll_item(zmq::POLLIN),
                self.request.as_poll_item(zmq::POLLIN),
            ];
            match zmq::poll(&mut items, 50) {
                Ok(0) | Err(zmq::Error::EINTR) => {}
                Ok(_) => idle_since = None,
                Err(zmq::Error::ETERM) => break,
                Err(error) => log_warn!(controller_id, "ZMQ poll error: {error}"),
            }
        }
        log_info!(controller_id, "stopped ZMQ worker");
        Ok(())
    }
}

// Decode and encode MessagePack frames outside the state mutex so the critical section only covers
// fast in-memory state transitions, and isolate unexpected panics so callers receive REQUEST_ERROR.
pub(crate) fn dispatch(
    controller_id: &str,
    state: &Mutex<ControllerState>,
    frames: &[Vec<u8>],
) -> Reply {
    let request = match wire::decode(frames) {
        Ok(request) => request,
        Err(error) => {
            log_error!(controller_id, "undecodable request: {error}");
            return Reply::error(None, format!("undecodable request: {error}"));
        }
    };
    let sender_id = request.sender_id.clone();
    let request_type = request.request_type.clone();
    let outcome = panic::catch_unwind(AssertUnwindSafe(|| execute_request(state, &request)));
    match outcome {
        Ok(Ok((kind, body))) => Reply {
            receiver: Some(sender_id),
            kind,
            body,
        },
        Ok(Err(error)) => {
            log_debug!(
                controller_id,
                "request_type={request_type} from sender={sender_id} failed: {error}"
            );
            Reply::error(Some(sender_id), format!("{request_type} failed: {error}"))
        }
        Err(payload) => {
            let detail = panic_message(payload.as_ref());
            log_error!(
                controller_id,
                "request_type={request_type} from sender={sender_id} panicked: {detail}"
            );
            Reply::error(
                Some(sender_id),
                format!("{request_type} panicked: {detail}"),
            )
        }
    }
}

fn execute_request(
    state: &Mutex<ControllerState>,
    request: &Request,
) -> std::result::Result<(&'static str, Body), String> {
    match request.request_type.as_str() {
        "SAVE_CONTROLLER_CHECKPOINT" => {
            let path = wire::text(&request.body, "path")?;
            ControllerState::save_checkpoint_unlocked(state, path)?;
            Ok((
                "SAVE_CONTROLLER_CHECKPOINT_RESPONSE",
                Body::Fields(vec![("success".into(), true.into())]),
            ))
        }
        "LOAD_CONTROLLER_CHECKPOINT" => {
            let path = wire::text(&request.body, "path")?;
            ControllerState::load_checkpoint_unlocked(state, path)?;
            Ok((
                "LOAD_CONTROLLER_CHECKPOINT_RESPONSE",
                Body::Fields(vec![("success".into(), true.into())]),
            ))
        }
        _ => lock_state(state)?.execute(request),
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.as_str()
    } else {
        "unknown internal panic"
    }
}

fn configured_max_sockets() -> Result<i32> {
    match std::env::var("TQ_CONTROLLER_ZMQ_MAX_SOCKETS") {
        Ok(raw) if !raw.is_empty() => {
            let parsed: i32 = raw.parse().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "TQ_CONTROLLER_ZMQ_MAX_SOCKETS must be a positive integer, got {raw:?}"
                    ),
                )
            })?;
            if parsed < 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("TQ_CONTROLLER_ZMQ_MAX_SOCKETS must be at least 1, got {parsed}"),
                )
                .into());
            }
            Ok(parsed)
        }
        _ => Ok(DEFAULT_ZMQ_MAX_SOCKETS),
    }
}

fn receive(socket: &zmq::Socket, controller_id: &str) -> Result<Option<Routed>> {
    loop {
        let frames = match socket.recv_multipart(zmq::DONTWAIT) {
            Ok(frames) => frames,
            Err(zmq::Error::EAGAIN | zmq::Error::EINTR) => return Ok(None),
            Err(zmq::Error::ETERM) => return Err(zmq::Error::ETERM.into()),
            Err(error) => {
                log_error!(controller_id, "ZMQ receive error: {error}");
                return Ok(None);
            }
        };
        let mut frames = frames.into_iter();
        if let Some(route) = frames.next() {
            return Ok(Some(Routed {
                route,
                data: frames.collect(),
            }));
        }
    }
}

fn send(socket: &zmq::Socket, route: &[u8], frame: &[u8], controller_id: &str) {
    if let Err(error) = socket.send_multipart([route, frame], 0) {
        log_error!(controller_id, "ZMQ send error: {error}");
    }
}

fn bind_router(context: &ZmqContext, ip: IpAddr) -> Result<(zmq::Socket, u16)> {
    let socket = context.router_socket()?;
    socket.set_linger(0)?;
    socket.set_rcvhwm(0)?;
    socket.set_sndhwm(0)?;
    socket.set_sndtimeo(50)?;
    let endpoint = if ip.is_ipv6() {
        socket.set_ipv6(true)?;
        format!("tcp://[{ip}]:*")
    } else {
        format!("tcp://{ip}:*")
    };
    socket.bind(&endpoint)?;
    let bound = socket
        .get_last_endpoint()?
        .map_err(|_| io::Error::other("ZeroMQ returned a non-UTF-8 endpoint"))?;
    let (_, port) = bound
        .rsplit_once(':')
        .ok_or_else(|| io::Error::other("ZeroMQ did not report a bound port"))?;
    Ok((socket, port.parse()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmpv::Value;
    use std::net::{IpAddr, Ipv4Addr};

    fn request_frame(request_type: &str, sender_id: &str, body: Value) -> Vec<u8> {
        let mut buf = Vec::new();
        rmpv::encode::write_value(
            &mut buf,
            &Value::Map(vec![
                ("request_id".into(), "req00001".into()),
                ("sender_id".into(), sender_id.into()),
                ("receiver_id".into(), Value::Nil),
                ("request_type".into(), request_type.into()),
                ("timestamp".into(), 1.5_f64.into()),
                ("body".into(), body),
            ]),
        )
        .unwrap();
        buf
    }

    #[test]
    fn sockets_raise_max_sockets_above_libzmq_default() {
        let sockets = Sockets::bind(IpAddr::V4(Ipv4Addr::LOCALHOST)).unwrap();
        assert!(sockets.max_sockets() > 1023);
        assert_eq!(sockets.max_sockets(), DEFAULT_ZMQ_MAX_SOCKETS as usize);
    }

    #[test]
    fn dispatch_recovers_from_poisoned_mutex_and_returns_request_error() {
        let state = Mutex::new(ControllerState::new(1, 1));
        let _ = panic::catch_unwind(AssertUnwindSafe(|| {
            let _guard = state.lock().unwrap();
            panic!("simulated invariant failure");
        }));
        assert!(state.is_poisoned());

        let frame = request_frame("GET_LIST_PARTITIONS", "client-1", Value::Map(vec![]));
        let reply = dispatch("ctrl-test", &state, &[frame]);
        assert_eq!(reply.kind, "REQUEST_ERROR");
        assert_eq!(reply.receiver.as_deref(), Some("client-1"));
        let Body::Fields(fields) = reply.body else {
            panic!("expected fields");
        };
        let message = wire::text(&fields, "message").unwrap();
        assert!(message.contains("poisoned"));
    }
}
