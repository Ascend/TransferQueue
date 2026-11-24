// Copyright 2026 The TransferQueue Team
// SPDX-License-Identifier: Apache-2.0

use std::hint;
use std::io;
use std::net::IpAddr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::Result;
use crate::state::ControllerState;
use crate::wire::{self, Reply};

const BATCH_DRAIN_LIMIT: usize = 256;
// Spin briefly before sleeping in zmq::poll so back-to-back ping-pong RPCs avoid kernel wakeups.
const IDLE_SPIN_WINDOW: Duration = Duration::from_micros(100);

struct Routed {
    route: Vec<u8>,
    data: Vec<Vec<u8>>,
}

pub(crate) struct Sockets {
    _context: zmq::Context,
    handshake: zmq::Socket,
    request: zmq::Socket,
    ports: (u16, u16),
}

impl Sockets {
    pub(crate) fn bind(ip: IpAddr) -> Result<Self> {
        let context = zmq::Context::new();
        let (handshake, handshake_port) = bind_router(&context, ip)?;
        let (request, request_port) = bind_router(&context, ip)?;
        Ok(Self {
            _context: context,
            handshake,
            request,
            ports: (handshake_port, request_port),
        })
    }

    pub(crate) fn ports(&self) -> (u16, u16) {
        self.ports
    }

    pub(crate) fn run(
        self,
        stopping: &AtomicBool,
        controller_id: &str,
        state: &Mutex<ControllerState>,
    ) -> Result<()> {
        let mut idle_since: Option<Instant> = None;
        while !stopping.load(Ordering::Relaxed) {
            let mut progressed = false;
            for _ in 0..BATCH_DRAIN_LIMIT {
                if stopping.load(Ordering::Relaxed) {
                    break;
                }
                let Some(message) = receive(&self.handshake)? else {
                    break;
                };
                progressed = true;
                match wire::handshake(&message.data).encode(controller_id) {
                    Ok(reply) => send(&self.handshake, &message.route, &reply),
                    Err(error) => eprintln!("Rust Controller handshake encode error: {error}"),
                }
            }
            for _ in 0..BATCH_DRAIN_LIMIT {
                if stopping.load(Ordering::Relaxed) {
                    break;
                }
                let Some(request) = receive(&self.request)? else {
                    break;
                };
                progressed = true;
                let reply = dispatch(state, &request.data);
                let receiver = reply.receiver.clone();
                let encoded = reply.encode(controller_id).or_else(|error| {
                    Reply::error(receiver, error.to_string()).encode(controller_id)
                });
                match encoded {
                    Ok(frame) => send(&self.request, &request.route, &frame),
                    Err(error) => eprintln!("Rust Controller reply encode error: {error}"),
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
                Err(error) => eprintln!("Rust Controller poll error: {error}"),
            }
        }
        Ok(())
    }
}

// Decode and encode MessagePack frames outside the state mutex so the critical section only covers
// fast in-memory state transitions.
fn dispatch(state: &Mutex<ControllerState>, frames: &[Vec<u8>]) -> Reply {
    let request = match wire::decode(frames) {
        Ok(request) => request,
        Err(error) => return Reply::error(None, error),
    };
    let result = state
        .lock()
        .expect("controller state poisoned")
        .execute(&request);
    match result {
        Ok((kind, body)) => Reply {
            receiver: Some(request.sender_id),
            kind,
            body,
        },
        Err(error) => Reply::error(Some(request.sender_id), error),
    }
}

fn receive(socket: &zmq::Socket) -> Result<Option<Routed>> {
    loop {
        let frames = match socket.recv_multipart(zmq::DONTWAIT) {
            Ok(frames) => frames,
            Err(zmq::Error::EAGAIN | zmq::Error::EINTR) => return Ok(None),
            Err(zmq::Error::ETERM) => return Err(zmq::Error::ETERM.into()),
            Err(error) => {
                eprintln!("Rust Controller receive error: {error}");
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

fn send(socket: &zmq::Socket, route: &[u8], frame: &[u8]) {
    if let Err(error) = socket.send_multipart([route, frame], 0) {
        eprintln!("Rust Controller send error: {error}");
    }
}

fn bind_router(context: &zmq::Context, ip: IpAddr) -> Result<(zmq::Socket, u16)> {
    let socket = context.socket(zmq::ROUTER)?;
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
