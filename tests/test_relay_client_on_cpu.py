# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest
import ray

from laminar.train.relay import ActorPublishStats, RelayBroadcastStats, RelayClient


def _broadcast_stats(*, logical_payload_bytes: int = 16) -> RelayBroadcastStats:
    goodput = float(logical_payload_bytes) / 0.25 if logical_payload_bytes else 0.0
    return RelayBroadcastStats(
        relay_broadcast_time_s=0.25,
        logical_payload_bytes=logical_payload_bytes,
        tensor_count=1,
        hop_count=1,
        bottleneck_hop_index=0,
        end_to_end_goodput_bytes_per_s=goodput,
    )


def _publisher_stats(
    *,
    payload_bytes: int = 16,
    transport_time_s: float = 0.25,
    sender_time_s: float = 0.5,
    total_time_s: float = 0.75,
) -> ActorPublishStats:
    return ActorPublishStats(
        publisher_count=1,
        payload_bytes=payload_bytes,
        sender_time_s=sender_time_s,
        transport_time_s=transport_time_s,
        total_time_s=total_time_s,
    )


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn
        self.calls: list[dict[str, Any]] = []

    def remote(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self._fn(*args, **kwargs)


class _RelayManagerActor:
    def __init__(self):
        self.trainer_version = -1
        self.loadable_version = -1
        self.active_publish: int | None = None
        self.begin_publish = _RemoteMethod(self._begin_publish)
        self.broadcast = _RemoteMethod(self._broadcast)
        self.complete_publish = _RemoteMethod(self._complete_publish)

    async def _begin_publish(self, *, version: int) -> None:
        self.trainer_version = max(self.trainer_version, version)
        self.active_publish = version

    async def _broadcast(
        self,
        *,
        version: int,
    ) -> RelayBroadcastStats:
        assert self.active_publish == version
        return _broadcast_stats()

    async def _complete_publish(self, *, version: int) -> None:
        assert self.active_publish == version
        self.loadable_version = max(self.loadable_version, version)
        self.active_publish = None


class _ActorWorkerGroup:
    def __init__(self):
        self.sent_weights: list[dict[str, Any]] = []
        self.wait_calls = 0

    def send_weights(
        self,
        *,
        version: int,
        dtype: str,
    ) -> ActorPublishStats:
        self.sent_weights.append(
            {
                "version": version,
                "dtype": dtype,
            }
        )
        return ActorPublishStats(publisher_count=1, payload_bytes=16)

    def wait_for_sending_to_relay(self) -> None:
        self.wait_calls += 1


def _ray_get(value):
    if isinstance(value, list):
        return [_ray_get(item) for item in value]
    if inspect.iscoroutine(value):
        return asyncio.run(value)
    return value


def test_relay_client_wires_core_manager_and_initialization_rpcs(monkeypatch):
    async def value(result):
        return result

    async def acquire(_node_ids, *, minimum_version):
        return minimum_version

    manager = SimpleNamespace(
        get_trainer_version=_RemoteMethod(lambda: value(3)),
        get_loadable_version=_RemoteMethod(lambda: value(2)),
        get_metrics=_RemoteMethod(lambda: value({"trainer_version": 3})),
        acquire_rollout_version=_RemoteMethod(acquire),
        release_full_read=_RemoteMethod(lambda _node_ids: value(None)),
        get_master_server_ip_port=_RemoteMethod(lambda: value(("127.0.0.1", 12345))),
        init_full_state_dict_in_shm=_RemoteMethod(lambda **_kwargs: "manager-init-ref"),
    )
    client = RelayClient(cast(Any, manager))
    monkeypatch.setattr(ray, "get", _ray_get)

    assert asyncio.run(client.get_trainer_version()) == 3
    assert asyncio.run(client.get_loadable_version()) == 2
    assert asyncio.run(client.get_metrics()) == {"trainer_version": 3}
    assert asyncio.run(client.acquire_rollout_version(["node-a"], minimum_version=2)) == 2
    assert asyncio.run(client.release_full_read_async(["node-a"])) is None
    actor_wg = SimpleNamespace(connect_to_relay_server=lambda **kwargs: setattr(actor_wg, "endpoint", kwargs))
    client.connect_actor_workers(actor_wg)
    state_meta = cast(Any, {"weight": object()})
    rollout_init = _RemoteMethod(lambda **_kwargs: "rollout-init-ref")
    rollout_server = SimpleNamespace(init_shm_state_dict_in_workers=rollout_init)

    assert client.init_full_state_dict_in_shm(full_state_meta_dict=state_meta) == "manager-init-ref"
    assert client.init_rollout_state_dict_in_shm(rollout_servers=[rollout_server], state_meta_dict=state_meta) == [
        "rollout-init-ref"
    ]
    assert actor_wg.endpoint == {"ip": "127.0.0.1", "port": 12345}
    assert manager.acquire_rollout_version.calls[-1] == {
        "args": (["node-a"],),
        "kwargs": {"minimum_version": 2},
    }
    assert manager.release_full_read.calls[-1] == {"args": (["node-a"],), "kwargs": {}}
    assert manager.init_full_state_dict_in_shm.calls[-1]["kwargs"] == {"full_state_meta_dict": state_meta}
    assert rollout_init.calls[-1]["kwargs"] == {"state_meta_dict": state_meta}


def test_relay_client_rejects_untyped_actor_publisher_stats():
    with pytest.raises(RuntimeError, match="Unexpected Relay actor send result: dict"):
        ActorPublishStats.from_results({"payload_bytes": 10})


def test_relay_client_fences_actor_send_after_send_ref_failure(monkeypatch):
    manager = _RelayManagerActor()
    client = RelayClient(cast(Any, manager))
    actor_wg = _ActorWorkerGroup()
    failed_ref = object()
    actor_wg.send_weights = lambda **_kwargs: failed_ref

    def materialize(value):
        if value is failed_ref:
            raise RuntimeError("actor send failed")
        return value

    monkeypatch.setattr(RelayClient, "_materialize_send_ref", staticmethod(materialize))

    with pytest.raises(RuntimeError, match="actor send failed"):
        client._send_actor_weights(actor_wg, version=7, dtype="bfloat16")

    assert actor_wg.wait_calls == 1


def test_relay_client_propagates_worker_send_failure_without_rechecking():
    class FenceWorkerGroup(_ActorWorkerGroup):
        def wait_for_sending_to_relay(self) -> None:
            self.wait_calls += 1
            raise RuntimeError("stored actor send failed")

    manager = _RelayManagerActor()
    client = RelayClient(cast(Any, manager))
    actor_wg = FenceWorkerGroup()
    actor_wg.send_weights = lambda **_kwargs: [None, None]

    with pytest.raises(RuntimeError, match="stored actor send failed"):
        asyncio.run(client.publish(actor_wg, version=7, dtype="bfloat16"))

    assert actor_wg.wait_calls == 1

    assert manager.complete_publish.calls == []


def test_relay_client_materializes_all_worker_refs_before_raising(monkeypatch):
    class FakeObjectRef:
        def __init__(self, value):
            self.value = value

    observed = []

    def get(ref):
        observed.append(ref.value)
        if isinstance(ref.value, BaseException):
            raise ref.value
        return ref.value

    monkeypatch.setattr(ray, "ObjectRef", FakeObjectRef)
    monkeypatch.setattr(ray, "get", get)

    refs = [FakeObjectRef(RuntimeError("rank 0 failed")), FakeObjectRef("rank 1 drained")]
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        RelayClient._materialize_send_ref(refs)

    assert observed == [refs[0].value, "rank 1 drained"]


def test_relay_client_publish_commits_after_both_barriers(monkeypatch):
    monkeypatch.setattr(ray, "get", _ray_get)
    manager = _RelayManagerActor()
    client = RelayClient(cast(Any, manager))
    actor_wg = _ActorWorkerGroup()
    actor_wg.send_weights = lambda **_kwargs: _publisher_stats()

    publish_metrics = asyncio.run(
        client.publish(
            actor_wg,
            version=7,
            dtype="bfloat16",
        )
    )

    assert manager.begin_publish.calls == [{"args": (), "kwargs": {"version": 7}}]
    assert manager.broadcast.calls == [{"args": (), "kwargs": {"version": 7}}]
    assert manager.complete_publish.calls == [{"args": (), "kwargs": {"version": 7}}]
    assert actor_wg.wait_calls == 0
    assert publish_metrics["relay_publish/completed_version"] == 7.0
    assert publish_metrics["relay_publish/actor_logical_payload_bytes"] == 16.0
    assert publish_metrics["relay/broadcast/logical_payload_bytes"] == 16.0
    assert publish_metrics["timing/relay_publish_total"] >= publish_metrics["timing/relay_publish_actor_flush"]
    assert publish_metrics["timing/relay_publish_total"] >= publish_metrics["timing/relay_publish_broadcast"]


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("payload_mismatch", "logical payload mismatch"),
        ("invalid_transport_goodput", "must be positive"),
    ],
)
def test_relay_client_publish_rejects_payload_or_protocol_mismatch_before_commit(case: str, error: str, monkeypatch):
    monkeypatch.setattr(ray, "get", _ray_get)
    manager = _RelayManagerActor()
    client = RelayClient(cast(Any, manager))
    actor_wg = _ActorWorkerGroup()
    actor_wg.send_weights = lambda **_kwargs: _publisher_stats(
        payload_bytes=32 if case == "payload_mismatch" else 16,
        transport_time_s=0.0 if case == "invalid_transport_goodput" else 0.25,
    )

    with pytest.raises(RuntimeError, match=error):
        asyncio.run(
            client.publish(
                actor_wg,
                version=7,
                dtype="bfloat16",
            )
        )
    assert manager.complete_publish.calls == []


def test_relay_client_publish_actor_send_does_not_block_event_loop(monkeypatch):
    monkeypatch.setattr(ray, "get", _ray_get)
    manager = _RelayManagerActor()
    broadcast_started = asyncio.Event()
    original_broadcast = manager._broadcast

    async def broadcast_and_signal(*, version: int):
        broadcast_started.set()
        return await original_broadcast(version=version)

    manager.broadcast = _RemoteMethod(broadcast_and_signal)
    client = RelayClient(cast(Any, manager))

    send_started = threading.Event()
    release_send = threading.Event()

    class BlockingWorkerGroup(_ActorWorkerGroup):
        def send_weights(self, **kwargs):
            del kwargs
            send_started.set()
            assert release_send.wait(timeout=1.0)
            return _publisher_stats()

    async def run():
        publish = asyncio.create_task(
            client.publish(
                BlockingWorkerGroup(),
                version=7,
                dtype="bfloat16",
            )
        )
        assert await asyncio.to_thread(send_started.wait, 1.0)
        await asyncio.wait_for(broadcast_started.wait(), timeout=1.0)
        assert not publish.done()
        release_send.set()
        await publish

    asyncio.run(run())


def test_relay_client_publish_actor_send_failure_fails_fast_without_retry(monkeypatch):
    monkeypatch.setattr(ray, "get", _ray_get)
    manager = _RelayManagerActor()
    broadcast_completed = threading.Event()
    original_broadcast = manager._broadcast

    async def broadcast_and_signal(*, version: int):
        broadcast_stats = await original_broadcast(version=version)
        broadcast_completed.set()
        return broadcast_stats

    manager.broadcast = _RemoteMethod(broadcast_and_signal)
    client = RelayClient(cast(Any, manager))

    class FailingWorkerGroup(_ActorWorkerGroup):
        def send_weights(self, **kwargs):
            del kwargs
            assert broadcast_completed.wait(timeout=1.0)
            raise RuntimeError("actor send failed")

    with pytest.raises(RuntimeError, match="actor send failed"):
        asyncio.run(
            client.publish(
                FailingWorkerGroup(),
                version=7,
                dtype="bfloat16",
            )
        )
    assert manager.begin_publish.calls
    assert manager.broadcast.calls
    assert manager.complete_publish.calls == []


def test_relay_client_publish_broadcast_failure_fails_fast_without_commit(monkeypatch):
    monkeypatch.setattr(ray, "get", _ray_get)
    manager = _RelayManagerActor()

    async def failing_broadcast(*, version: int):
        del version
        raise RuntimeError("relay broadcast failed")

    manager.broadcast = _RemoteMethod(failing_broadcast)
    client = RelayClient(cast(Any, manager))
    actor_wg = _ActorWorkerGroup()
    actor_wg.send_weights = lambda **_kwargs: _publisher_stats()

    with pytest.raises(RuntimeError, match="relay broadcast failed"):
        asyncio.run(
            client.publish(
                actor_wg,
                version=7,
                dtype="bfloat16",
            )
        )
    assert manager.begin_publish.calls
    assert manager.broadcast.calls
    assert manager.complete_publish.calls == []


def test_relay_client_publish_drain_preserves_concurrent_broadcast_failure(monkeypatch):
    monkeypatch.setattr(ray, "get", _ray_get)
    manager = _RelayManagerActor()
    broadcast_started = threading.Event()

    async def failing_broadcast(*, version: int):
        del version
        broadcast_started.set()
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass
        raise RuntimeError("broadcast independently failed during drain")

    manager.broadcast = _RemoteMethod(failing_broadcast)
    client = RelayClient(cast(Any, manager))

    class RacingWorkerGroup(_ActorWorkerGroup):
        def send_weights(self, **kwargs):
            del kwargs
            assert broadcast_started.wait(timeout=1.0)
            raise RuntimeError("actor send failed first")

    with pytest.raises(RuntimeError, match="actor send failed first"):
        asyncio.run(
            client.publish(
                RacingWorkerGroup(),
                version=7,
                dtype="bfloat16",
            )
        )
    assert len(manager.begin_publish.calls) == 1
    assert manager.complete_publish.calls == []


def test_relay_client_publish_cancellation_drains_child_tasks(monkeypatch):
    monkeypatch.setattr(ray, "get", _ray_get)

    async def run():
        manager = _RelayManagerActor()
        broadcast_started = asyncio.Event()
        broadcast_stopped = asyncio.Event()

        async def blocking_broadcast(*, version: int):
            del version
            broadcast_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                broadcast_stopped.set()

        manager.broadcast = _RemoteMethod(blocking_broadcast)
        client = RelayClient(cast(Any, manager))
        actor_wg = _ActorWorkerGroup()
        actor_wg.send_weights = lambda **_kwargs: _publisher_stats()

        publish_task = asyncio.create_task(
            client.publish(
                actor_wg,
                version=7,
                dtype="bfloat16",
            )
        )
        await broadcast_started.wait()
        publish_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await publish_task

        assert broadcast_stopped.is_set()
        assert manager.complete_publish.calls == []

    asyncio.run(run())


@pytest.mark.parametrize("send_result", [None, [None, None]])
def test_relay_client_background_send_waits_for_future_stats(send_result):
    class BackgroundWorkerGroup(_ActorWorkerGroup):
        def send_weights(self, **kwargs):
            return send_result

        def wait_for_sending_to_relay(self):
            self.wait_calls += 1
            return [None, _publisher_stats()]

    worker_group = BackgroundWorkerGroup()
    client = RelayClient(cast(Any, _RelayManagerActor()))
    assert client._send_actor_weights(worker_group, version=7, dtype="bfloat16") == _publisher_stats()
    assert worker_group.wait_calls == 1


def test_relay_client_completed_send_commits_while_next_compute_is_blocked(monkeypatch):
    monkeypatch.setattr(ray, "get", _ray_get)
    compute_started = threading.Event()
    release_compute = threading.Event()

    def next_compute():
        compute_started.set()
        assert release_compute.wait(timeout=2.0)

    compute_thread = threading.Thread(target=next_compute)

    class CompletedWorkerGroup(_ActorWorkerGroup):
        def send_weights(self, **kwargs):
            compute_thread.start()
            assert compute_started.wait(timeout=1.0)
            return [None, _publisher_stats()]

        def wait_for_sending_to_relay(self):
            self.wait_calls += 1
            # A redundant RPC cannot run while the next synchronous compute
            # occupies the actor. Successful send results must avoid this RPC.
            assert release_compute.wait(timeout=1.0)

    worker_group = CompletedWorkerGroup()
    manager = _RelayManagerActor()
    client = RelayClient(cast(Any, manager))
    try:
        result = asyncio.run(client.publish(worker_group, version=7, dtype="bfloat16"))
        assert compute_started.is_set() and compute_thread.is_alive()
        assert not release_compute.is_set()
        assert worker_group.wait_calls == 0
        assert result["relay_publish/completed_version"] == 7
        assert len(manager.complete_publish.calls) == 1
    finally:
        release_compute.set()
        compute_thread.join(timeout=2.0)


def test_relay_client_rank_send_failure_drains_every_ref_and_does_not_commit(monkeypatch):
    class FakeObjectRef:
        def __init__(self, value):
            self.value = value

    failure = RuntimeError("rank 1 send failed")
    drain_failure = RuntimeError("rank 1 drain failed")
    observed = []

    def get(ref):
        observed.append(ref.value)
        if isinstance(ref.value, BaseException):
            raise ref.value
        return ref.value

    monkeypatch.setattr(ray, "ObjectRef", FakeObjectRef)
    monkeypatch.setattr(ray, "get", get)

    class FailedWorkerGroup(_ActorWorkerGroup):
        def send_weights(self, **kwargs):
            return [FakeObjectRef(_publisher_stats()), FakeObjectRef(failure), FakeObjectRef(None)]

        def wait_for_sending_to_relay(self):
            self.wait_calls += 1
            return [FakeObjectRef(None), FakeObjectRef(drain_failure), FakeObjectRef(None)]

    manager = _RelayManagerActor()
    client = RelayClient(cast(Any, manager))
    worker_group = FailedWorkerGroup()
    with pytest.raises(RuntimeError, match="rank 1 send failed") as caught:
        asyncio.run(client.publish(worker_group, version=7, dtype="bfloat16"))
    assert caught.value is failure
    assert observed == [_publisher_stats(), failure, None, None, drain_failure, None]
    assert worker_group.wait_calls == 1
    assert manager.complete_publish.calls == []
    assert any("rank 1 drain failed" in note for note in failure.__notes__)
