# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from laminar.train import main as main_async_rl


class _RemoteMethod:
    def __init__(self, ref: object):
        self.ref = ref
        self.calls = []

    def remote(self, *args, **kwargs) -> object:
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.ref


class _Actor:
    def __init__(
        self,
        *,
        start_ref: object | None = None,
        refresh_ref: object | None = None,
        fit_ref: object | None = None,
        wait_ref: object | None = None,
        prepare_ref: object | None = None,
    ):
        if start_ref is not None:
            self.start = _RemoteMethod(start_ref)
        if refresh_ref is not None:
            self.refresh_once = _RemoteMethod(refresh_ref)
        if fit_ref is not None:
            self.fit = _RemoteMethod(fit_ref)
        if wait_ref is not None:
            self.wait_for_stop = _RemoteMethod(wait_ref)
        if prepare_ref is not None or fit_ref is not None:
            self.prepare = _RemoteMethod("trainer.prepare" if prepare_ref is None else prepare_ref)


def _rollout_dispatcher_actor() -> _Actor:
    return _Actor(start_ref="rollout.start", wait_ref="rollout.wait")


def _llm_server_state_tracker_actor() -> _Actor:
    return _Actor(
        start_ref="state_tracker.start",
        refresh_ref="state_tracker.refresh",
        wait_ref="state_tracker.wait",
    )


def test_runtime_restores_trainer_before_rollout_dispatcher_starts(monkeypatch):
    runtime = main_async_rl._AsyncRLRuntime(
        trainer=_Actor(prepare_ref="trainer.prepare", fit_ref="trainer.fit"),
        rollout_dispatcher=_rollout_dispatcher_actor(),
        llm_server_state_tracker=_llm_server_state_tracker_actor(),
        multi_version_rollout_manager=_Actor(
            start_ref="multi.start",
            wait_ref="multi.wait",
        ),
    )
    calls = []

    def fake_ray_get(refs):  # noqa: ANN001, ANN202
        calls.append(refs)
        return None

    def fake_ray_wait(refs, *, num_returns):  # noqa: ANN001, ANN202
        assert num_returns == 1
        return ["trainer.fit"], [ref for ref in refs if ref != "trainer.fit"]

    monkeypatch.setattr(main_async_rl.ray, "get", fake_ray_get)
    monkeypatch.setattr(main_async_rl.ray, "wait", fake_ray_wait)

    runtime.run()

    assert calls == [
        "state_tracker.refresh",
        "state_tracker.start",
        "multi.start",
        "trainer.prepare",
        "rollout.start",
        "trainer.fit",
    ]


class _Runtime:
    def __init__(self, exc: BaseException | None = None):
        self.exc = exc

    def run(self) -> None:
        if self.exc is not None:
            raise self.exc


def test_runtime_fails_when_multi_version_manager_loop_fails(monkeypatch):
    trainer = _Actor(fit_ref="trainer.fit")
    rollout_dispatcher = _rollout_dispatcher_actor()
    llm_server_state_tracker = _llm_server_state_tracker_actor()
    multi_version_rollout_manager = _Actor(
        start_ref="multi.start",
        wait_ref="multi.wait",
    )
    runtime = main_async_rl._AsyncRLRuntime(
        trainer=trainer,
        rollout_dispatcher=rollout_dispatcher,
        llm_server_state_tracker=llm_server_state_tracker,
        multi_version_rollout_manager=multi_version_rollout_manager,
    )

    def fake_ray_get(refs):  # noqa: ANN001, ANN202
        if refs == "multi.start":
            return True
        if refs == "multi.wait":
            raise RuntimeError("repack load failed")
        return None

    def fake_ray_wait(refs, *, num_returns):  # noqa: ANN001, ANN202
        assert refs == ["trainer.fit", "state_tracker.wait", "rollout.wait", "multi.wait"]
        assert num_returns == 1
        return ["multi.wait"], ["trainer.fit", "state_tracker.wait", "rollout.wait"]

    monkeypatch.setattr(main_async_rl.ray, "get", fake_ray_get)
    monkeypatch.setattr(main_async_rl.ray, "wait", fake_ray_wait)

    with pytest.raises(RuntimeError, match="repack load failed"):
        runtime.run()


@pytest.mark.parametrize(
    ("failed_ref", "error_message"),
    [
        ("state_tracker.wait", "snapshot poll failed"),
        ("rollout.wait", "rollout dispatch failed"),
    ],
)
def test_runtime_fails_when_required_control_loop_fails(monkeypatch, failed_ref, error_message):
    runtime = main_async_rl._AsyncRLRuntime(
        trainer=_Actor(fit_ref="trainer.fit"),
        rollout_dispatcher=_rollout_dispatcher_actor(),
        llm_server_state_tracker=_llm_server_state_tracker_actor(),
        multi_version_rollout_manager=_Actor(
            start_ref="multi.start",
            wait_ref="multi.wait",
        ),
    )

    def fake_ray_get(refs):  # noqa: ANN001, ANN202
        if refs == failed_ref:
            raise RuntimeError(error_message)
        return None

    def fake_ray_wait(refs, *, num_returns):  # noqa: ANN001, ANN202
        assert refs == ["trainer.fit", "state_tracker.wait", "rollout.wait", "multi.wait"]
        assert num_returns == 1
        return [failed_ref], [ref for ref in refs if ref != failed_ref]

    monkeypatch.setattr(main_async_rl.ray, "get", fake_ray_get)
    monkeypatch.setattr(main_async_rl.ray, "wait", fake_ray_wait)

    with pytest.raises(RuntimeError, match=error_message):
        runtime.run()


def test_runtime_raises_if_multi_version_manager_stops_cleanly_before_trainer(monkeypatch):
    trainer = _Actor(fit_ref="trainer.fit")
    rollout_dispatcher = _rollout_dispatcher_actor()
    llm_server_state_tracker = _llm_server_state_tracker_actor()
    multi_version_rollout_manager = _Actor(
        start_ref="multi.start",
        wait_ref="multi.wait",
    )
    runtime = main_async_rl._AsyncRLRuntime(
        trainer=trainer,
        rollout_dispatcher=rollout_dispatcher,
        llm_server_state_tracker=llm_server_state_tracker,
        multi_version_rollout_manager=multi_version_rollout_manager,
    )

    def fake_ray_get(refs):  # noqa: ANN001, ANN202
        return refs == "multi.start"

    def fake_ray_wait(refs, *, num_returns):  # noqa: ANN001, ANN202
        assert refs == ["trainer.fit", "state_tracker.wait", "rollout.wait", "multi.wait"]
        assert num_returns == 1
        return ["multi.wait"], ["trainer.fit", "state_tracker.wait", "rollout.wait"]

    monkeypatch.setattr(main_async_rl.ray, "get", fake_ray_get)
    monkeypatch.setattr(main_async_rl.ray, "wait", fake_ray_wait)

    with pytest.raises(RuntimeError, match="multi-version rollout manager stopped before trainer.fit completed"):
        runtime.run()


def test_main_task_inits_and_closes_transfer_queue(monkeypatch):
    calls: list[tuple[str, object]] = []

    class _FakeTQ:
        def init(self, config):  # noqa: ANN001
            calls.append(("init", config))

        def close(self) -> None:
            calls.append(("close", None))

    monkeypatch.setattr(main_async_rl, "tq", _FakeTQ())
    monkeypatch.setattr(main_async_rl, "_initialize_runtime", lambda _config: _Runtime())
    config = SimpleNamespace(transfer_queue=SimpleNamespace())

    main_async_rl.main_task._function(config)

    assert calls == [("init", config.transfer_queue), ("close", None)]


def test_main_task_closes_transfer_queue_after_runtime_failure(monkeypatch):
    calls: list[str] = []

    class _FakeTQ:
        def init(self, config):  # noqa: ANN001
            calls.append("init")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(main_async_rl, "tq", _FakeTQ())
    monkeypatch.setattr(main_async_rl, "_initialize_runtime", lambda _config: _Runtime(ValueError("trainer failed")))

    with pytest.raises(ValueError, match="trainer failed"):
        main_async_rl.main_task._function(SimpleNamespace(transfer_queue=SimpleNamespace()))

    assert calls == ["init", "close"]


def _runtime_with_tracking():
    tracker = _Actor()
    tracker.finish = _RemoteMethod("tracker.finish")
    runtime = main_async_rl._AsyncRLRuntime(
        trainer=_Actor(fit_ref="trainer.fit"),
        rollout_dispatcher=_rollout_dispatcher_actor(),
        llm_server_state_tracker=_llm_server_state_tracker_actor(),
        multi_version_rollout_manager=_Actor(start_ref="multi.start", wait_ref="multi.wait"),
        tracker=tracker,
    )
    return runtime, tracker


def test_runtime_waits_for_successful_tracker_finish_after_fit(monkeypatch):
    runtime, tracker = _runtime_with_tracking()
    calls = []
    monkeypatch.setattr(main_async_rl.ray, "get", lambda ref: calls.append(ref))
    monkeypatch.setattr(main_async_rl.ray, "wait", lambda _refs, **_kwargs: (["trainer.fit"], []))
    runtime.run()
    assert calls[-2:] == ["trainer.fit", "tracker.finish"]
    assert tracker.finish.calls == [{"args": (), "kwargs": {"exit_code": 0}}]


def test_runtime_cleanup_failure_does_not_mask_original_training_error(monkeypatch):
    runtime, tracker = _runtime_with_tracking()
    original = ValueError("original trainer failure")

    def get(ref):
        if ref == "trainer.fit":
            raise original
        if ref == "tracker.finish":
            raise OSError("backend close failure")

    monkeypatch.setattr(main_async_rl.ray, "get", get)
    monkeypatch.setattr(main_async_rl.ray, "wait", lambda _refs, **_kwargs: (["trainer.fit"], []))
    with pytest.raises(ValueError) as caught:
        runtime.run()
    assert caught.value is original
    assert tracker.finish.calls == [{"args": (), "kwargs": {"exit_code": 1}}]


def test_runtime_setup_failure_closes_existing_tracker_with_failure_status(monkeypatch):
    runtime, tracker = _runtime_with_tracking()

    def get(ref):
        if ref == "trainer.prepare":
            raise ValueError("restore failed")

    monkeypatch.setattr(main_async_rl.ray, "get", get)
    with pytest.raises(ValueError, match="restore failed"):
        runtime.run()
    assert tracker.finish.calls == [{"args": (), "kwargs": {"exit_code": 1}}]


def test_runtime_successful_fit_does_not_hide_tracker_finalization_failure(monkeypatch):
    runtime, tracker = _runtime_with_tracking()
    original = OSError("finalization failed")

    def get(ref):
        if ref == "tracker.finish":
            raise original

    monkeypatch.setattr(main_async_rl.ray, "get", get)
    monkeypatch.setattr(main_async_rl.ray, "wait", lambda _refs, **_kwargs: (["trainer.fit"], []))
    with pytest.raises(OSError) as caught:
        runtime.run()
    assert caught.value is original
    assert [call["kwargs"]["exit_code"] for call in tracker.finish.calls] == [0, 1]
