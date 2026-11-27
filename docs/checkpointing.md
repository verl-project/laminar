# Asynchronous per-step checkpoints

Megatron trainers can overlap restart-checkpoint writes with the next training
step. Enable the native distributed checkpoint format and keep a bounded number
of committed checkpoints:

```bash
bash recipes/swe/train.sh \
  trainer.async_checkpoint=true \
  trainer.save_freq=1 \
  trainer.max_actor_ckpt_to_keep=2 \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=true
```

`trainer.async_checkpoint` is the Laminar switch and defaults to `false`
for synchronous checkpointing. The actor's `checkpoint.async_save` setting
is derived from it. At model initialization the worker enables the shared
writer through verl's `enable_trainer_owned_threaded_checkpointing` manager
method. Async saving requires a verl build that provides this API and Megatron's
ordinary distributed optimizer and model path, with `model`, `optimizer`, and
`extra` in both checkpoint save and load contents. Unsupported worker
configurations include FSDP, Megatron-FSDP, PEFT, and HF model export, and fail
before saving.

The native format setting controls restart saves and loads independently of the
initial model source. With the default `dist_checkpointing_path=null`, actor
initialization loads `actor_rollout_ref.model.path` through the existing HF
bridge, then creates a native checkpoint manager. An explicit
`actor_rollout_ref.actor.megatron.dist_checkpointing_path` continues to select
native model weights for initialization. Full training recovery still uses
`trainer.resume_mode` and restores model, optimizer, scheduler, and RNG state
from the selected restart checkpoint.

Each save stages independent model, optimizer, scheduler, and RNG state on the
training ranks before returning. A dedicated I/O thread operates on that snapshot;
later optimizer updates cannot change its tensors. The dispatcher separately
saves its prompt-dataloader cursor in `data.pt`, following the existing restart
contract. Buffered and in-flight trajectories are not replayed on restart.

There is at most one pending checkpoint per trainer. The next checkpoint first
finalizes the preceding actor save on all ranks and waits for `data.pt`.
Finalization writes the actor manifest. The trainer owns storage, publication
and retention. It atomically replaces `latest_checkpointed_iteration.txt` only
after both saves complete, then prunes older complete checkpoints and records
`checkpoint_completed`. A failed write does not advance the marker or delete
its previous target. The normal final training step flushes the pending
checkpoint before finalizing metrics.

The latest marker can lag the optimizer by up to one checkpoint interval plus
the current step: a finished background write is committed at the next save
boundary. Backpressure blocks that boundary if storage cannot keep up. Each
rank retains at most one pending checkpoint snapshot. Total process memory
also depends on the training workload, allocator and duration of the write.
The feature does not remove the storage bandwidth needed to write every step.
The checkpoint root must be shared and durable for every training rank; an
actor-only `trainer.default_hdfs_dir` remains unsupported.

Restart checkpoints do not require a matching Relay publication. The configured
`trainer.weight_publish_interval` stays independent, so `save_freq=1` does not
force rollout weight updates every step. Colocated training stages before
handing GPUs back to rollout, while detached CPU writes can continue.

Use `trainer.resume_mode=auto` with the same checkpoint root to restore the
committed marker target, or `trainer.resume_mode=resume_path` and
`trainer.resume_from_path=/shared/checkpoints/global_step_N` for an explicitly
verified checkpoint. Model, optimizer, scheduler, RNG, and dispatcher state use
the existing verl loader and layout. These native restart checkpoints do not
contain HF weights for `laminar.eval`; produce HF exports separately when
offline evaluation requires them.

For an async-enabled Megatron actor, the restart loader releases unused CUDA
allocator blocks after native loading. This gives NCCL room for the first
forward after restart. The cleanup runs once at load time.

When changing an existing run to the native distributed format, use a new
checkpoint root or select a verified compatible checkpoint explicitly. These
flags do not convert checkpoints previously written in a different format.

## Reproduce

With the [Megatron development environment](../README.md#getting-started)
and the `dev` extra, run the checkpoint and configuration tests. These checks
use CPU fixtures and require no training dataset or checkpoint storage service.

```bash
CUDA_VISIBLE_DEVICES='' uv run --locked --extra dev --extra megatron \
  --extra vllm --extra ucx python -m pytest -q \
  tests/test_async_train_shell_on_cpu.py \
  tests/test_async_checkpoint_on_cpu.py \
  tests/test_native_checkpoint_bootstrap_on_cpu.py \
  tests/test_per_step_checkpoint_on_cpu.py
```

For a short training smoke test, prepare the [public SWE recipe](../recipes/swe/README.md#prerequisites)
and use a new checkpoint directory shared by the trainer and dispatcher.

```bash
checkpoint_dir=/shared/checkpoints/laminar-async-smoke
uv run --locked --extra megatron --extra vllm --extra ucx --extra swe \
  bash recipes/swe/train.sh \
  trainer.total_training_steps=4 \
  trainer.async_checkpoint=true \
  trainer.save_freq=1 \
  trainer.max_actor_ckpt_to_keep=2 \
  "trainer.default_local_dir=${checkpoint_dir}" \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=true

test "$(cat "${checkpoint_dir}/latest_checkpointed_iteration.txt")" = 4
test -s "${checkpoint_dir}/global_step_4/data.pt"
test -s "${checkpoint_dir}/global_step_4/actor/ckpt_contents.json"
```

## Measuring overhead

- `checkpoint/stage_time_s`: foreground staging RPC and dispatcher scheduling.
- `checkpoint/backpressure_time_s`: completion of the preceding checkpoint at
  a new save boundary, including marker and retention work.
- `checkpoint/finalize_wait_time_s`: residual all-rank/cursor wait plus commit
  work, recorded in the checkpoint's completion event.
- `checkpoint/worker_snapshot_bytes` and `checkpoint/worker_cpu_snapshot_copy_bytes`:
  sums of the rank-local staged payload and CPU isolation copies. Worker stage
  and finalize times use the maximum across ranks, rather than summing nested
  or concurrent waits. Snapshot bytes do not measure peak process or GPU memory.
- `checkpoint/save_time_s`: elapsed lifecycle from staging to commit, including
  overlapped training and time until the next save boundary. It is not isolated
  disk-write time.
- `training_cycle_completed` events include staging and backpressure tails;
  immutable optimizer-step rows retain their existing timing scope.

Compare checkpointing disabled, native synchronous saving and native
asynchronous saving using the same updates, snapshot contents and storage.
For foreground overhead, divide mean in-loop checkpoint time by the no-save
mean compute time. Use `checkpoint/save_time_s` for synchronous saves and
`checkpoint/stage_time_s` plus `checkpoint/backpressure_time_s` for asynchronous
saves. These intervals already include in-loop retention. Retain post-loop
drain and cleanup separately. Use full wall time including these tails to
measure throughput through the final durable checkpoint.

## Limiting interference with training

Laminar uses verl's threaded Megatron checkpoint manager. Its background writer
reuses MCore's serialization, fsync and completion callbacks, and feeds each
writer's results directly to its finalizer through an in-process queue. MCore
can initialize its native results sidecar once during planning. The background
writer creates no separate CUDA context and does not send checkpoint tensors
between processes. Each content
tree (`model`, `optimizer`, `extra`) validates sharding access integrity on its
first save, then caches its parallel save distribution and PyTorch distributed
checkpoint save plan across steps. GPU staging stays on the actor RPC thread and
finishes before training can mutate parameters. All distributed completion
callbacks also stay on that RPC thread; the I/O thread receives staged CPU
tensors and serialized objects only.

[PyTorch's storage write releases the GIL](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/torch/csrc/jit/python/init.cpp#L1483-L1493),
allowing training to continue during large writes. Python metadata serialization and shared memory/storage bandwidth
can still affect training. The implementation does not change the training
process's CPU or I/O priority. A Future failure is synchronized across ranks
before completion callbacks can advance the latest marker.

[MCore's native CPU writer](https://github.com/NVIDIA/Megatron-LM/blob/4b4acac9a1d28ea6829c8d4f566d75698a21249d/megatron/core/dist_checkpointing/strategies/filesystem_async.py#L252)
temporarily disables process-wide automatic cyclic GC
and restores its prior enabled state on completion or exception. Explicit
`gc.collect()` remains available. The single pending checkpoint bounds this
interval to the active write; long storage stalls can delay automatic collection.
This does not bound the duration of a stalled write or total cyclic-object memory.
Normal shutdown drains the writer. An active native file write cannot be safely
cancelled independently of its owning actor.
