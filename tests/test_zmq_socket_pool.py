# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ZMQSocketPool's concurrency invariants.

Sockets used to be created and closed per request. The pool reuses them, which is only
safe because a lease is exclusive and a socket that did not complete a clean send/recv is
discarded: replies carry no request id (see ZMQMessage.create), so a reply left in flight
by a timed-out or cancelled request would be read by the next user of that socket as its
own. test_timed_out_socket_is_not_reused pins exactly that.

Each role's own reuse is asserted where that role is tested -- the client's controller RPC
in test_client.py, the metrics collector in test_metrics.py, pool wiring and context
lifecycle in test_zmq_shared_context.py. What is left here needs control over reply timing
and loop lifetime that a caller-level test cannot reach.

Reuse is asserted from the peer's side rather than from the pool's internals: every socket
dials with its own ZMQ identity, so one identity across many requests means the connection
was reused, and a fresh identity means the old socket was discarded.
"""

import asyncio
import threading
from unittest.mock import patch

import pytest
import zmq
import zmq.asyncio

import transfer_queue.utils.zmq_utils as zmq_utils
from transfer_queue.utils.enum_utils import Role
from transfer_queue.utils.zmq_utils import ZMQServerInfo, ZMQSocketPool


class _Peer:
    """A ROUTER that echoes one reply per request, recording who dialled it.

    ``tag`` distinguishes which endpoint answered, for the re-registration test.
    """

    def __init__(self, delay_first_reply: float = 0.0, peer_id: str = "peer_0", tag: bytes = b"reply-to-"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.info = ZMQServerInfo(role=Role.STORAGE, id=peer_id, ip="127.0.0.1", ports={"put_get_socket": port})
        self.identities: list[bytes] = []
        self._tag = tag
        self._delay_first_reply = delay_first_reply
        self._replies = 0
        self.running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    @property
    def callers(self) -> int:
        """How many distinct sockets have dialled this peer."""
        return len(set(self.identities))

    def _serve(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while self.running:
            if not dict(poller.poll(50)):
                continue
            identity, request = self.socket.recv_multipart()
            self.identities.append(identity)
            if self._replies == 0 and self._delay_first_reply:
                # Reply late enough that the requester has already timed out, leaving this
                # reply in flight -- the poisoned-socket scenario.
                time_left = self._delay_first_reply
                while time_left > 0 and self.running:
                    sleep = min(0.05, time_left)
                    threading.Event().wait(sleep)
                    time_left -= sleep
            self._replies += 1
            self.socket.send_multipart([identity, self._tag + request])

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.socket.close(linger=0)
        self.context.term()


@pytest.fixture
def peer():
    p = _Peer()
    yield p
    p.stop()


async def _round_trip(pool, peer_info, payload=b"req"):
    with pool.lease(peer_info) as sock:
        await sock.send_multipart([payload])
        return (await sock.recv_multipart())[0]


@pytest.mark.asyncio
async def test_timed_out_socket_is_not_reused():
    """A timed-out request must not leave its socket -- or its late reply -- in the pool.

    Regression guard for the core hazard: with the socket pooled, the *next* request would
    receive the previous request's reply, silently attributing one response to another.
    """
    # Replies to the first request only after the pool's 1s timeout has expired, so that
    # reply is still in flight when the socket would otherwise go to the next caller. The
    # peer serves serially, so the delay stays well inside the second request's own timeout.
    peer = _Peer(delay_first_reply=1.4)
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner", "put_get_socket", timeout=1)
    try:
        with pytest.raises(zmq.error.Again):
            await _round_trip(pool, peer.info, b"first")

        # The late reply to "first" must not surface as the answer to "second".
        assert await _round_trip(pool, peer.info, b"second") == b"reply-to-second"
        assert peer.callers == 2, "the timed-out socket was handed to the next request"
    finally:
        pool.close()
        ctx.destroy(linger=0)
        peer.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "cancellation"])
async def test_poisoned_lease_is_discarded(failure):
    """A lease that did not complete cleanly must not return its socket to the pool.

    The request is on the wire before the failure and its reply is still in flight, so
    parking the socket would hand the next caller a reply belonging to someone else.
    Cancellation counts as well as a raise: asyncio.gather cancels its siblings on the
    first failure, so this is the routine case rather than an exotic one.
    """
    # Answers the first request only after it has been abandoned, which is what leaves the
    # stale reply in flight.
    peer = _Peer(delay_first_reply=0.6)
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner", "put_get_socket")

    async def poisoned():
        with pool.lease(peer.info) as sock:
            await sock.send_multipart([b"doomed"])
            # Fail only once the peer has the request, so its reply really is outstanding.
            while not peer.identities:
                await asyncio.sleep(0.01)
            if failure == "exception":
                raise RuntimeError("handler blew up")
            await asyncio.sleep(60)  # cancelled here, after the lease was handed out

    try:
        if failure == "exception":
            with pytest.raises(RuntimeError):
                await poisoned()
        else:
            task = asyncio.create_task(poisoned())
            while not peer.identities:
                await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        abandoned = set(peer.identities)
        # The reply to "doomed" must not surface as the answer to "second".
        assert await _round_trip(pool, peer.info, b"second") == b"reply-to-second"
        assert set(peer.identities) - abandoned, "the poisoned socket was reused"
    finally:
        pool.close()
        ctx.destroy(linger=0)
        peer.stop()


def test_sockets_are_not_reused_across_event_loops(peer):
    """A socket bound to a finished loop must never be handed to another one.

    pyzmq rebinds an async socket to whatever loop it next sees; one bound to a *closed*
    loop is the "Bad file descriptor / SIGABRT" failure this keying exists to prevent.
    """
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner", "put_get_socket")

    async def lease_twice(tag):
        # Twice per loop, so a socket IS reused within a loop -- which is what makes the
        # cross-loop comparison meaningful rather than trivially true.
        for i in range(2):
            assert await _round_trip(pool, peer.info, f"{tag}{i}".encode()) == f"reply-to-{tag}{i}".encode()

    for tag in ("a", "b", "c"):
        asyncio.run(lease_twice(tag))

    assert peer.callers == 3, "one socket per loop, reused within it"

    pool.close()
    ctx.destroy(linger=0)


@pytest.mark.asyncio
async def test_reregistered_peer_is_not_served_a_stale_socket():
    """A peer that moves to a new address must not be answered by its old endpoint.

    A storage unit or controller restarted or re-registered keeps its id but gets a fresh
    port, so keying reuse on the id alone would keep leasing a socket wired to the address
    it no longer answers on -- silently talking to a dead or reassigned endpoint. All calls
    share one event loop, as a real client or storage manager does; a loop per call would
    discard the socket via owner eviction and hide the bug.
    """
    old = _Peer(peer_id="su0", tag=b"OLD:")
    new = _Peer(peer_id="su0", tag=b"NEW:")
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner", "put_get_socket")
    try:
        assert await _round_trip(pool, old.info) == b"OLD:req"
        # Same id, different port -- exactly what re-registration produces.
        assert await _round_trip(pool, new.info) == b"NEW:req"
        # And a peer that moves back is still reachable.
        assert await _round_trip(pool, old.info) == b"OLD:req"
    finally:
        pool.close()
        ctx.destroy(linger=0)
        old.stop()
        new.stop()


@pytest.mark.asyncio
async def test_burst_beyond_pool_size_is_served(peer):
    """Concurrency above maxsize must not be refused or blocked, only left unparked."""
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner", "put_get_socket", maxsize=2)

    results = await asyncio.gather(*[_round_trip(pool, peer.info, f"c{i}".encode()) for i in range(8)])
    assert sorted(results) == sorted(f"reply-to-c{i}".encode() for i in range(8))

    pool.close()
    ctx.destroy(linger=0)


@pytest.mark.asyncio
async def test_alease_caps_sockets_in_flight(peer):
    """alease waits for a socket rather than opening one past maxsize.

    This is what makes the context's socket budget a function of configuration: without it
    the peak equals real concurrency, which at a few thousand peers exhausts ZMQ_MAX_SOCKETS
    and fails a lease with EMFILE.
    """
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner", "put_get_socket", maxsize=4)

    async def one(i):
        async with pool.alease(peer.info) as sock:
            await sock.send_multipart([f"c{i}".encode()])
            return (await sock.recv_multipart())[0]

    results = await asyncio.gather(*[one(i) for i in range(40)])

    assert sorted(results) == sorted(f"reply-to-c{i}".encode() for i in range(40)), "a request was dropped"
    assert peer.callers == 4, "concurrency past maxsize opened extra sockets instead of waiting"

    pool.close()
    ctx.destroy(linger=0)


@pytest.mark.asyncio
async def test_alease_permits_are_per_address(peer):
    """A busy peer must not stall requests to a different one.

    Permits are keyed like the buckets, so a fan-out across N peers is not serialized by a
    cap meant to bound one peer's concurrency.
    """
    other = _Peer(peer_id="peer_1", tag=b"other-")
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner", "put_get_socket", maxsize=1)

    async def hold():
        async with pool.alease(peer.info) as sock:
            await sock.send_multipart([b"slow"])
            await sock.recv_multipart()
            await asyncio.sleep(0.3)  # keeps peer's only permit

    async def other_peer():
        async with pool.alease(other.info) as sock:
            await sock.send_multipart([b"req"])
            return (await sock.recv_multipart())[0]

    # Would time out if one peer's permit gated the other.
    _, reply = await asyncio.wait_for(asyncio.gather(hold(), other_peer()), timeout=5)
    assert reply == b"other-req"

    pool.close()
    ctx.destroy(linger=0)
    other.stop()


@pytest.mark.asyncio
async def test_nested_alease_raises_instead_of_hanging(peer):
    """A lease inside another lease must fail loudly rather than wait on a held permit.

    A permit is released when its body ends, so nesting can wait on one the same task --
    or a sibling mid-cycle -- already holds. That hangs with no timeout and no error, the
    failure mode hardest to diagnose in production.
    """
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner", "put_get_socket", maxsize=8)

    with pytest.raises(RuntimeError, match="already holding"):
        async with pool.alease(peer.info):
            async with pool.alease(peer.info):
                pass

    # The outer permit is returned, so the pool still works.
    async with pool.alease(peer.info) as sock:
        await sock.send_multipart([b"after"])
        assert (await sock.recv_multipart())[0] == b"reply-to-after"

    pool.close()
    ctx.destroy(linger=0)


@pytest.mark.asyncio
async def test_nesting_is_rejected_across_pools(peer):
    """Two pools do not make nesting safe: a cycle between them deadlocks just as well.

    Each task holds what the other waits for, and separate semaphores do not break that.
    """
    other = _Peer(peer_id="peer_1", tag=b"other-")
    ctx = zmq.asyncio.Context()
    first = ZMQSocketPool(ctx, "first", "put_get_socket", maxsize=1)
    second = ZMQSocketPool(ctx, "second", "put_get_socket", maxsize=1)

    with pytest.raises(RuntimeError, match="already holding"):
        async with first.alease(peer.info):
            async with second.alease(other.info):
                pass

    first.close()
    second.close()
    ctx.destroy(linger=0)
    other.stop()


@pytest.mark.asyncio
async def test_consecutive_leases_are_not_nesting(peer):
    """Leasing again after the previous body ended is the supported shape."""
    ctx = zmq.asyncio.Context()
    pool = ZMQSocketPool(ctx, "owner", "put_get_socket", maxsize=2)

    for i in range(3):
        async with pool.alease(peer.info) as sock:
            await sock.send_multipart([f"s{i}".encode()])
            assert (await sock.recv_multipart())[0] == f"reply-to-s{i}".encode()

    pool.close()
    ctx.destroy(linger=0)


def test_pool_size_comes_from_the_env_var_at_construction():
    """Every role's pool takes TQ_SOCKET_POOL_SIZE, resolved per pool rather than at import.

    A default bound in the signature would freeze whatever the environment held when this
    module first loaded, which is what makes such a knob look settable but do nothing.
    """
    ctx = zmq.Context()
    try:
        with patch.object(zmq_utils, "TQ_SOCKET_POOL_SIZE", 7):
            assert ZMQSocketPool(ctx, "owner", "put_get_socket")._maxsize == 7
            # An explicit argument still wins, which is what lets a caller opt out.
            assert ZMQSocketPool(ctx, "owner", "put_get_socket", maxsize=3)._maxsize == 3
    finally:
        ctx.destroy(linger=0)


def test_pool_size_below_one_is_rejected():
    """Below 1 nothing is ever parked, so reuse is off while the pool still looks pooled."""
    ctx = zmq.Context()
    try:
        with patch.object(zmq_utils, "TQ_SOCKET_POOL_SIZE", 0):
            with pytest.raises(ValueError, match="at least 1"):
                ZMQSocketPool(ctx, "owner", "put_get_socket")
    finally:
        ctx.destroy(linger=0)
