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

"""
A simplified data-centric RL workflow demo built with StreamingDataset and
StreamingDataLoader.

The implementation structure and asynchronous dataflow are inspired by the
Relax project, while keeping the example intentionally lightweight and focused
on educational readability. Reference: https://github.com/redai-infra/Relax
"""

import argparse
import os
import time
from dataclasses import dataclass
from importlib import resources

# Disable Ray's cross-worker log deduplication before importing Ray itself,
# otherwise many worker-side prints will be folded into "[repeated Nx across cluster]".
os.environ["RAY_DEDUP_LOGS"] = "0"

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

import transfer_queue as tq
from transfer_queue import RankAwareSampler, StreamingDataLoader, StreamingDataset
from transfer_queue.metadata import BatchMeta
from transfer_queue.utils.logging_utils import get_logger

logger = get_logger("MAIN", default_level="INFO")

STAGE_NAMES = ["rollout", "ref", "actor", "reward", "update"]


def emit_worker_log(message: str, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def make_prompt_batch(step: int, config: "DemoConfig") -> TensorDict:
    start_id = step * config.global_batch_size
    generator = torch.Generator().manual_seed(config.seed + step)
    sample_ids = torch.arange(start_id, start_id + config.global_batch_size, dtype=torch.long)
    prompt_ids = torch.randint(
        0,
        config.vocab_size,
        (config.global_batch_size, config.prompt_length),
        generator=generator,
        dtype=torch.long,
    )
    return TensorDict({"sample_id": sample_ids.unsqueeze(-1), "prompt_ids": prompt_ids}, batch_size=config.global_batch_size)


def generate_sequences(prompt_ids: torch.Tensor, config: "DemoConfig") -> TensorDict:
    batch_size = prompt_ids.size(0)
    generator = torch.Generator().manual_seed(config.seed + int(prompt_ids.sum().item()))
    response_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, config.response_length),
        generator=generator,
        dtype=torch.long,
    )
    return TensorDict(
        {
            "input_ids": torch.cat([prompt_ids, response_ids], dim=1),
            "response_ids": response_ids,
            "response_mask": torch.ones_like(response_ids),
        },
        batch_size=batch_size,
    )


def compute_log_prob(prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> torch.Tensor:
    return (response_ids.float().mean(dim=1, keepdim=True) - prompt_ids.float().mean(dim=1, keepdim=True)) / 1000.0


def compute_reward(response_ids: torch.Tensor) -> torch.Tensor:
    return response_ids.float().mean(dim=1, keepdim=True) / 1000.0


def compute_loss(old_log_prob: torch.Tensor, ref_log_prob: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    return (old_log_prob - ref_log_prob - advantage).abs()


@ray.remote
class ProgressTracker:
    def __init__(self, stage_names: list[str], num_steps: int):
        self.counts = {stage: {step: 0 for step in range(num_steps)} for stage in stage_names}
        self.done_workers = {stage: {step: 0 for step in range(num_steps)} for stage in stage_names}

    def record(self, stage: str, step: int, batch_size: int) -> int:
        self.counts[stage][step] = self.counts.get(stage, {}).get(step, 0) + batch_size
        return self.counts[stage][step]

    def record_done(self, stage: str, step: int) -> int:
        self.done_workers[stage][step] = self.done_workers.get(stage, {}).get(step, 0) + 1
        return self.done_workers[stage][step]

    def get_counts(self, step: int) -> dict:
        return {stage: self.counts[stage].get(step, 0) for stage in self.counts}

    def get_done_workers(self, step: int) -> dict:
        return {stage: self.done_workers[stage].get(step, 0) for stage in self.done_workers}


class BaseStageWorker:
    stage_name = "base"

    def __init__(self, tq_config, tracker, worker_id: int, config: "DemoConfig"):
        tq.init(tq_config)
        self.tq_client = tq.get_client()
        controller = ray.get_actor("TransferQueueController")
        self.cfg = ray.get(controller.get_config.remote())
        self.tracker = tracker
        self.worker_id = worker_id
        self.cfg_demo = config
        self.worker_name = f"{self.stage_name}-{worker_id}"
        self._dataloader: StreamingDataLoader | None = None

    def start(self, iteration: int, train_iters: int) -> dict:
        for step in range(iteration, train_iters):
            self._run_step(step)
        return {"worker": self.worker_name, "stage": self.stage_name}

    def _run_step(self, step: int) -> None:
        partition_id = f"{self.cfg_demo.partition_prefix}_{step}"
        dataloader = self._get_dataloader(partition_id)

        for batch, batch_meta in dataloader:
            sample_ids = batch["sample_id"].view(-1).tolist()
            emit_worker_log(
                f"[{self.worker_name}] step={step} consume: sample_ids={sample_ids}",
                self.cfg_demo.enable_worker_logs,
            )

            output, written_fields = self.compute(batch, batch_meta)
            self.tq_client.put(output, metadata=batch_meta)

            count = ray.get(self.tracker.record.remote(self.stage_name, step, len(sample_ids)))
            emit_worker_log(
                f"[{self.worker_name}] step={step} produce: "
                f"fields={written_fields}, count={count}/{self.cfg_demo.global_batch_size}",
                self.cfg_demo.enable_worker_logs,
            )

        ray.get(self.tracker.record_done.remote(self.stage_name, step))

    def _get_dataloader(self, partition_id: str) -> StreamingDataLoader:
        if self._dataloader is None:
            self._dataloader = self._build_dataloader(partition_id)
        else:
            self._dataloader.step(partition_id)
        return self._dataloader

    def _build_dataloader(self, partition_id: str) -> StreamingDataLoader:
        dataset = StreamingDataset(
            config=self.cfg,
            batch_size=self.cfg_demo.micro_batch_size,
            micro_batch_size=self.cfg_demo.micro_batch_size,
            data_fields=self.input_fields(),
            partition_id=partition_id,
            task_name=f"{self.cfg_demo.task_name_prefix}_{self.stage_name}",
            dp_rank=self.worker_id,
            should_check_consumption_status=True,
        )
        return StreamingDataLoader(dataset=dataset, num_workers=0, prefetch_factor=None)

    def input_fields(self) -> list[str]:
        raise NotImplementedError

    def base_sleep_seconds(self) -> float:
        return self.cfg_demo.stage_sleep_seconds

    def compute(self, batch: TensorDict, batch_meta: BatchMeta):
        raise NotImplementedError

    def sleep_with_jitter(self, batch_meta: BatchMeta) -> None:
        jitter_seed = self.worker_id * 7 + int(batch_meta.global_indexes[0])
        jitter = 0.05 * (jitter_seed % 5)
        time.sleep(max(0.0, self.base_sleep_seconds() + jitter))


@ray.remote(num_cpus=0.1)
class RolloutWorker(BaseStageWorker):
    stage_name = "rollout"

    def input_fields(self) -> list[str]:
        return ["sample_id", "prompt_ids"]

    def base_sleep_seconds(self) -> float:
        return self.cfg_demo.rollout_sleep_seconds

    def compute(self, batch: TensorDict, batch_meta: BatchMeta):
        self.sleep_with_jitter(batch_meta)
        output = generate_sequences(batch["prompt_ids"], self.cfg_demo)
        return output, ["input_ids", "response_ids", "response_mask"]


@ray.remote(num_cpus=0.1)
class RefWorker(BaseStageWorker):
    stage_name = "ref"

    def input_fields(self) -> list[str]:
        return ["sample_id", "prompt_ids", "response_ids"]

    def compute(self, batch: TensorDict, batch_meta: BatchMeta):
        self.sleep_with_jitter(batch_meta)
        log_prob = compute_log_prob(batch["prompt_ids"], batch["response_ids"])
        return TensorDict({"ref_log_prob": log_prob}, batch_size=log_prob.size(0)), ["ref_log_prob"]


@ray.remote(num_cpus=0.1)
class ActorWorker(BaseStageWorker):
    stage_name = "actor"

    def input_fields(self) -> list[str]:
        return ["sample_id", "prompt_ids", "response_ids"]

    def compute(self, batch: TensorDict, batch_meta: BatchMeta):
        self.sleep_with_jitter(batch_meta)
        log_prob = compute_log_prob(batch["prompt_ids"], batch["response_ids"])
        return TensorDict({"old_log_prob": log_prob}, batch_size=log_prob.size(0)), ["old_log_prob"]


@ray.remote(num_cpus=0.1)
class RewardWorker(BaseStageWorker):
    stage_name = "reward"

    def input_fields(self) -> list[str]:
        return ["sample_id", "response_ids"]

    def compute(self, batch: TensorDict, batch_meta: BatchMeta):
        self.sleep_with_jitter(batch_meta)
        advantage = compute_reward(batch["response_ids"])
        return TensorDict({"advantage": advantage}, batch_size=advantage.size(0)), ["advantage"]


@ray.remote(num_cpus=0.1)
class UpdateWorker(BaseStageWorker):
    stage_name = "update"

    def input_fields(self) -> list[str]:
        return ["sample_id", "old_log_prob", "ref_log_prob", "advantage"]

    def compute(self, batch: TensorDict, batch_meta: BatchMeta):
        self.sleep_with_jitter(batch_meta)
        loss = compute_loss(batch["old_log_prob"], batch["ref_log_prob"], batch["advantage"])
        return TensorDict({"loss": loss}, batch_size=loss.size(0)), ["loss"]


@ray.remote(num_cpus=0.1)
def sync_weights(step: int, sleep_s: float) -> dict:
    time.sleep(sleep_s)
    return {"step": step}


@dataclass(frozen=True)
class DemoConfig:
    partition_prefix: str
    task_name_prefix: str
    num_steps: int
    global_batch_size: int
    micro_batch_size: int
    prompt_length: int
    response_length: int
    vocab_size: int
    num_rollout_workers: int
    num_ref_workers: int
    num_actor_workers: int
    num_reward_workers: int
    num_update_workers: int
    rollout_sleep_seconds: float
    stage_sleep_seconds: float
    weight_sync_seconds: float
    num_data_storage_units: int
    seed: int
    enable_worker_logs: bool

    def validate(self) -> None:
        for name, value in [
            ("num_steps", self.num_steps),
            ("global_batch_size", self.global_batch_size),
            ("micro_batch_size", self.micro_batch_size),
            ("prompt_length", self.prompt_length),
            ("response_length", self.response_length),
            ("vocab_size", self.vocab_size),
            ("num_rollout_workers", self.num_rollout_workers),
            ("num_ref_workers", self.num_ref_workers),
            ("num_actor_workers", self.num_actor_workers),
            ("num_reward_workers", self.num_reward_workers),
            ("num_update_workers", self.num_update_workers),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.global_batch_size % self.micro_batch_size != 0:
            raise ValueError("global_batch_size % micro_batch_size != 0")


def build_tq_config(config: DemoConfig):
    base = OmegaConf.load(resources.files("transfer_queue") / "config.yaml")
    override = OmegaConf.create(
        {
            "controller": {"sampler": RankAwareSampler, "polling_mode": True},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {"num_data_storage_units": config.num_data_storage_units},
            },
        },
        flags={"allow_objects": True},
    )
    return OmegaConf.merge(base, override)


class DataCentricPipelineDemo:
    def __init__(self, config: DemoConfig, tq_config):
        self.config = config
        tq.init(tq_config)
        self.tq_client = tq.get_client()
        self.tracker = ProgressTracker.remote(STAGE_NAMES, config.num_steps)

        self.rollout_workers = [RolloutWorker.remote(tq_config, self.tracker, i, config) for i in range(config.num_rollout_workers)]
        self.ref_workers = [RefWorker.remote(tq_config, self.tracker, i, config) for i in range(config.num_ref_workers)]
        self.actor_workers = [ActorWorker.remote(tq_config, self.tracker, i, config) for i in range(config.num_actor_workers)]
        self.reward_workers = [RewardWorker.remote(tq_config, self.tracker, i, config) for i in range(config.num_reward_workers)]
        self.update_workers = [UpdateWorker.remote(tq_config, self.tracker, i, config) for i in range(config.num_update_workers)]

    def _put_prompt(self, step: int) -> None:
        partition_id = f"{self.config.partition_prefix}_{step}"
        batch = make_prompt_batch(step, self.config)
        sample_ids = batch["sample_id"].view(-1).tolist()
        meta = self.tq_client.put(batch, partition_id=partition_id)
        logger.info(
            f"step={step} | put prompts: "
            f"partition={partition_id}, sample_ids={sample_ids}, fields={list(meta.field_names)}"
        )

    def _wait_complete(self, step: int) -> None:
        while True:
            counts = ray.get(self.tracker.get_counts.remote(step))
            done_workers = ray.get(self.tracker.get_done_workers.remote(step))

            active_counts = {stage: count for stage, count in counts.items() if count > 0}
            logger.info(f"step={step} | progress: counts={active_counts}, done_workers={done_workers}")

            all_workers_done = (
                done_workers.get("rollout", 0) >= self.config.num_rollout_workers
                and done_workers.get("ref", 0) >= self.config.num_ref_workers
                and done_workers.get("actor", 0) >= self.config.num_actor_workers
                and done_workers.get("reward", 0) >= self.config.num_reward_workers
                and done_workers.get("update", 0) >= self.config.num_update_workers
            )
            if all_workers_done:
                return
            time.sleep(0.2)

    def _start_worker_group(self, workers: list) -> list:
        return [worker.start.remote(0, self.config.num_steps) for worker in workers]

    def fit(self) -> list[dict]:
        logger.info("=" * 72)
        logger.info("TransferQueue StreamingDataLoader Data-Centric Pipeline Demo (Relax-inspired)")
        logger.info("=" * 72)
        logger.info(
            f"workers | rollout={self.config.num_rollout_workers}, "
            f"ref={self.config.num_ref_workers}, actor={self.config.num_actor_workers}, "
            f"reward={self.config.num_reward_workers}, update={self.config.num_update_workers}"
        )
        logger.info(
            f"pipeline | num_steps={self.config.num_steps}, "
            f"global_batch_size={self.config.global_batch_size}, "
            f"micro_batch_size={self.config.micro_batch_size}"
        )

        refs = []
        refs.extend(self._start_worker_group(self.rollout_workers))
        refs.extend(self._start_worker_group(self.ref_workers))
        refs.extend(self._start_worker_group(self.actor_workers))
        refs.extend(self._start_worker_group(self.reward_workers))
        refs.extend(self._start_worker_group(self.update_workers))

        for step in range(self.config.num_steps):
            logger.info("=" * 72)
            logger.info(f"STEP {step}")
            logger.info("=" * 72)
            self._put_prompt(step)
            self._wait_complete(step)
            logger.info(f"step={step} | weight sync: start")
            ray.get(sync_weights.remote(step, self.config.weight_sync_seconds))
            logger.info(f"step={step} | weight sync: done")
            self.tq_client.clear_partition(f"{self.config.partition_prefix}_{step}")
            logger.info(f"step={step} | clear partition: {self.config.partition_prefix}_{step}")

        ray.get(refs)
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-prefix", type=str, default="decentralized_demo")
    parser.add_argument("--task-name-prefix", type=str, default="decentralized")
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--prompt-length", type=int, default=24)
    parser.add_argument("--response-length", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--num-rollout-workers", type=int, default=2)
    parser.add_argument("--num-ref-workers", type=int, default=2)
    parser.add_argument("--num-actor-workers", type=int, default=2)
    parser.add_argument("--num-reward-workers", type=int, default=2)
    parser.add_argument("--num-update-workers", type=int, default=1)
    parser.add_argument("--rollout-sleep-seconds", type=float, default=0.30)
    parser.add_argument("--stage-sleep-seconds", type=float, default=0.15)
    parser.add_argument("--weight-sync-seconds", type=float, default=0.20)
    parser.add_argument("--num-data-storage-units", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260410)
    parser.add_argument("--enable-worker-logs", action="store_true")
    args = parser.parse_args()

    cfg = DemoConfig(
        partition_prefix=args.partition_prefix,
        task_name_prefix=args.task_name_prefix,
        num_steps=args.num_steps,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        prompt_length=args.prompt_length,
        response_length=args.response_length,
        vocab_size=args.vocab_size,
        num_rollout_workers=args.num_rollout_workers,
        num_ref_workers=args.num_ref_workers,
        num_actor_workers=args.num_actor_workers,
        num_reward_workers=args.num_reward_workers,
        num_update_workers=args.num_update_workers,
        rollout_sleep_seconds=args.rollout_sleep_seconds,
        stage_sleep_seconds=args.stage_sleep_seconds,
        weight_sync_seconds=args.weight_sync_seconds,
        num_data_storage_units=args.num_data_storage_units,
        seed=args.seed,
        enable_worker_logs=args.enable_worker_logs,
    )
    cfg.validate()

    os.environ["TQ_PRE_ALLOC_SAMPLE_NUM"] = str(cfg.global_batch_size)

    completed = False
    ray.init()
    try:
        demo = DataCentricPipelineDemo(cfg, build_tq_config(cfg))
        demo.fit()
        completed = True
    finally:
        tq.close()
        ray.shutdown()

    if completed:
        logger.info("demo done!")


if __name__ == "__main__":
    main()
