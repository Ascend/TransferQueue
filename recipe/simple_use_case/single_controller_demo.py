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

import argparse
import asyncio
import logging
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack
from torch.utils.data import DataLoader, Dataset

import transfer_queue as tq
from transfer_queue import KVBatchMeta

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DEBUG"] = "1"


def compute_log_prob(data1, _data2):
    print(f"compute_log_prob: data1 {data1}, data2 {_data2}")
    time.sleep(3)
    return _data2


def compute_loss(data1, _data2):
    time.sleep(3)
    return data1


def compute_reward(response_ids: torch.Tensor) -> TensorDict:
    """Simulate a reward model that scores each token position in the response.

    Returns a TensorDict with an ``"advantage"`` field whose shape matches
    ``response_ids`` (i.e. one scalar per response token).
    """
    time.sleep(1)
    advantage = torch.randn_like(response_ids, dtype=torch.float32)
    return TensorDict({"advantage": advantage}, batch_size=response_ids.size(0))


class TrainingWorker:
    def __init__(self, role):
        self.role = role

    def train_mini_batch(self, kv_meta: KVBatchMeta) -> KVBatchMeta:
        """Simulate multi-mini-batch training loop"""

        assert self.role == "actor"

        # 1. Pull data from storage
        data = tq.kv_batch_get_by_meta(meta=kv_meta)
        logger.info(f"train_mini_batch: got data {data}")

        # 2. Compute loss
        output = compute_loss(data["old_log_prob"], data["ref_log_prob"])
        output = TensorDict({"loss": output}, batch_size=output.size(0))

        # 3. Write back
        kv_meta = tq.kv_batch_put(keys=kv_meta.keys, partition_id=kv_meta.partition_id, fields=output)
        logger.info("train_mini_batch: put data done")

        return kv_meta

    def infer_batch(self, kv_meta: KVBatchMeta) -> KVBatchMeta:
        """Simulate forward-only inference"""
        # 1. Pull data from storage
        data = tq.kv_batch_get_by_meta(meta=kv_meta)
        logger.info(f"compute_log_prob: got data {data}")

        # 2. Model forward
        output = compute_log_prob(data["prompt_ids"], data["response_ids"])
        if self.role == "actor":
            output = TensorDict({"old_log_prob": output}, batch_size=output.size(0))
        elif self.role == "ref":
            output = TensorDict({"ref_log_prob": output}, batch_size=output.size(0))
        else:
            raise ValueError(f"Role {self.role} not supported.")

        # 3. Write back
        kv_meta = tq.kv_batch_put(keys=kv_meta.keys, partition_id=kv_meta.partition_id, fields=output)
        logger.info("infer_batch: put data done")

        return kv_meta


class ActorRolloutRefWorker:
    def __init__(self):
        self.actor = TrainingWorker(role="actor")
        self.ref = TrainingWorker(role="ref")

    def compute_ref_log_prob(self, kv_meta: KVBatchMeta) -> KVBatchMeta:
        output = self.ref.infer_batch(kv_meta)
        return output

    def compute_log_prob(self, kv_meta: KVBatchMeta) -> KVBatchMeta:
        output = self.actor.infer_batch(kv_meta)
        return output

    def update_actor(self, kv_meta: KVBatchMeta) -> KVBatchMeta:
        output = self.actor.train_mini_batch(kv_meta)
        return output

    async def update_weights(self, global_steps: int = None):
        # Simulate weight sync from actor to rollout
        logger.info(f"update_weights: syncing weights at step {global_steps}")
        await asyncio.sleep(1)


async def generate(prompt: torch.Tensor, response_length: int, vocab_size: int) -> torch.Tensor:
    assert prompt.ndim == 1
    response = torch.randint(low=0, high=vocab_size, size=(response_length,), dtype=torch.long)
    return response


IMAGE_TOKEN_ID = 32001


def simulate_chat_template(messages: list[dict], vocab_size: int, image_token_length: int = 64) -> torch.Tensor:
    """Simulate ``tokenizer.apply_chat_template`` with interleaved image support.

    Each message follows the OpenAI-style multi-modal format::

        {"role": "user",
         "content": [
             {"type": "image_url", "image_url": {"url": "..."}},
             {"type": "text", "text": "Describe this image"},
         ]}

    ``content`` may also be a plain string for text-only messages.

    - ``"text"`` parts are tokenised as one random ID per whitespace word.
    - ``"image_url"`` parts each produce ``image_token_length`` placeholder
      tokens (simulating the patch embeddings a vision encoder would emit).

    Args:
        messages: Chat-style message list.
        vocab_size: Vocabulary size for random text token IDs.
        image_token_length: Number of placeholder tokens per image.

    Returns:
        1-D ``torch.Tensor`` of token IDs.
    """
    tokens: list[int] = []
    for msg in messages:
        content = msg.get("content", "")

        if isinstance(content, str):
            if content:
                tokens.extend(torch.randint(0, vocab_size, (len(content.split()),)).tolist())
        elif isinstance(content, list):
            for part in content:
                part_type = part.get("type", "")
                if part_type == "text":
                    text = part.get("text", "")
                    if text:
                        tokens.extend(torch.randint(0, vocab_size, (len(text.split()),)).tolist())
                elif part_type == "image_url":
                    tokens.extend([IMAGE_TOKEN_ID] * image_token_length)

    return torch.tensor(tokens, dtype=torch.long)


@dataclass
class MessageDatasetConfig:
    """Configuration for :class:`MessageDataset`."""

    num_samples: int = 1000
    text_length_range: tuple[int, int] = (10, 128)
    vocab_size: int = 32000
    num_images_range: tuple[int, int] = (0, 3)


class MessageDataset(Dataset):
    """Dataset that yields OpenAI-style messages with random-length text.

    Each sample is a dict containing a ``"messages"`` key with the message
    list.  Text length is sampled uniformly from ``text_length_range`` and
    the number of images per message is sampled from ``num_images_range``.
    """

    def __init__(self, config: MessageDatasetConfig):
        self.config = config

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, idx: int) -> dict:
        cfg = self.config
        text_len = random.randint(*cfg.text_length_range)
        num_images = random.randint(*cfg.num_images_range)

        words = [str(random.randint(0, cfg.vocab_size - 1)) for _ in range(text_len)]
        text = " ".join(words)

        content: list[dict] = []
        for _ in range(num_images):
            content.append({"type": "image_url", "image_url": {"url": "simulated"}})
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]
        return {"messages": messages}


def message_collate_fn(batch: list[dict]) -> TensorDict:
    """Collate a batch of message dicts into a ``TensorDict``.

    Each sample's ``"messages"`` list is stored as a ``NonTensorStack``
    entry so that the entire batch can be represented as a single
    ``TensorDict`` with ``batch_size == len(batch)``.
    """
    messages_list = [sample["messages"] for sample in batch]
    return TensorDict(
        {"messages": NonTensorStack(*messages_list)},
        batch_size=len(batch),
    )


@dataclass
class AgentLoopConfig:
    """Configuration for :class:`AgentLoop` multi-turn rollout."""

    max_turns_range: tuple[int, int] = (1, 4)
    tool_response_length_range: tuple[int, int] = (5, 20)
    vocab_size: int = 32000
    response_length: int = 32
    image_token_length: int = 64


class AgentLoop:
    """Multi-turn agentic rollout that interleaves LLM generation with tool calls.

    Each turn:
      1. Call ``generate()`` to produce a model response.
      2. Check whether the response triggers a tool call.
      3. If yes, simulate tool execution and append the tool-response tokens.
      4. Repeat until no tool call is detected or ``max_turns`` is reached.
    """

    def __init__(self, config: AgentLoopConfig):
        self.config = config

    async def run(self, data: TensorDict) -> TensorDict:
        """Execute a multi-turn rollout for a single sample.

        Args:
            data: ``TensorDict`` with ``batch_size=1``.  Must contain a
                ``"messages"`` field (stored via ``NonTensorStack``) holding
                an OpenAI-style message list, e.g.::

                    [{"role": "user",
                      "content": [
                          {"type": "image_url",
                           "image_url": {"url": "https://...jpg"}},
                          {"type": "text",
                           "text": "Describe this image"},
                      ]}]

        Returns:
            ``TensorDict`` with ``batch_size=1`` containing:

            - ``"input_ids"`` — concatenation of prompt and response,
              shape ``[1, prompt_len + response_len]``.
            - ``"prompt_ids"`` — token IDs of the original message, shape
              ``[1, prompt_len]``.
            - ``"response_ids"`` — all generated tokens (generations + tool
              responses across every turn), shape ``[1, response_len]``.
            - ``"response_mask"`` — ``1`` for model-generated tokens,
              ``0`` for tool-response tokens, shape ``[1, response_len]``.
            - ``"num_turns"`` — how many generation turns were executed,
              shape ``[1]``.
        """
        cfg = self.config
        min_turns, max_turns = cfg.max_turns_range
        num_turns = random.randint(min_turns, max_turns)

        assert data.batch_size[0] == 1, "batch_size must be 1"

        messages = list(data["messages"])[0]
        prompt = simulate_chat_template(messages, cfg.vocab_size, cfg.image_token_length)
        logger.info(
            f"AgentLoop: initial prompt length = {prompt.shape[0]}, "
            f"sampled {num_turns} turns (range {cfg.max_turns_range})"
        )

        conversation = prompt.clone()
        response_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []

        for turn in range(num_turns):
            gen = await generate(conversation, cfg.response_length, cfg.vocab_size)
            conversation = torch.cat([conversation, gen])
            response_parts.append(gen)
            mask_parts.append(torch.ones(gen.shape[0], dtype=torch.long))
            logger.info(
                f"AgentLoop turn {turn}/{num_turns}: generated {gen.shape[0]} tokens, "
                f"conversation length = {conversation.shape[0]}"
            )

            if not self._detect_tool_call(turn, num_turns):
                logger.info(f"AgentLoop turn {turn}: final answer produced, rollout complete.")
                break

            tool_response = self._simulate_tool_response()
            conversation = torch.cat([conversation, tool_response])
            response_parts.append(tool_response)
            mask_parts.append(torch.zeros(tool_response.shape[0], dtype=torch.long))
            logger.info(
                f"AgentLoop turn {turn}: tool call → appended {tool_response.shape[0]} "
                f"tool-response tokens, conversation length = {conversation.shape[0]}"
            )

        response = torch.cat(response_parts) if response_parts else torch.tensor([], dtype=torch.long)
        response_mask = torch.cat(mask_parts) if mask_parts else torch.tensor([], dtype=torch.long)
        input_ids = torch.cat([prompt, response])

        data = TensorDict(
            {
                "input_ids": input_ids.unsqueeze(0),
                "prompt_ids": prompt.unsqueeze(0),
                "response_ids": response.unsqueeze(0),
                "response_mask": response_mask.unsqueeze(0),
                "num_turns": torch.tensor([turn + 1]),
            },
            batch_size=1,
        )
        return data

    def _detect_tool_call(self, turn: int, num_turns: int) -> bool:
        """Simulate tool-call detection.

        In a real agent this would parse the decoded model output for
        tool-call syntax (e.g. function-call JSON).  Here we
        deterministically issue a tool call on every turn except the last
        one, guaranteeing multi-turn behaviour in the demo.
        """
        return turn < num_turns - 1

    def _simulate_tool_response(self) -> torch.Tensor:
        """Simulate tool execution returning random token IDs.

        The response length is sampled uniformly from
        ``[tool_response_length_range[0], tool_response_length_range[1]]``.
        """
        min_len, max_len = self.config.tool_response_length_range
        length = random.randint(min_len, max_len)
        return torch.randint(0, self.config.vocab_size, (length,), dtype=torch.long)


@ray.remote(num_cpus=1)
class AgentLoopWorker:
    def __init__(self, tq_config, agent_loop_config: AgentLoopConfig):
        tq.init(tq_config)
        self.agent_loop_config = agent_loop_config

    async def generate_sequences(self, kv_meta_chunk):
        print(f"demo get data -> generate_sequences {kv_meta_chunk}")

        # chunk the kv_meta_chunk into a list of kv_meta and create an agentloop for each kv_meta
        kv_meta_chunks = kv_meta_chunk.chunk(len(kv_meta_chunk))
        tasks = []
        for kv_meta in kv_meta_chunks:
            tasks.append(asyncio.create_task(self.generate(kv_meta)))
        kv_metas = await asyncio.gather(*tasks)
        return KVBatchMeta.concat(kv_metas)

    async def generate(self, kv_meta):
        data = tq.kv_batch_get_by_meta(meta=kv_meta)
        agent_loop = AgentLoop(config=self.agent_loop_config)
        output = await agent_loop.run(data)
        kv_meta_new = tq.kv_batch_put(keys=kv_meta.keys, partition_id=kv_meta.partition_id, fields=output)
        print(f"demo put data -> generate {kv_meta_new}")
        return kv_meta_new


class AgentLoopManager:
    def __init__(self, num_workers: int, agent_loop_config: AgentLoopConfig, tq_config):
        tq.init(tq_config)

        self.async_rollout_workers = []
        for _ in range(num_workers):
            self.async_rollout_workers.append(AgentLoopWorker.remote(tq_config, agent_loop_config))

    def generate_sequences(self, kv_meta):
        kv_meta_chunks = kv_meta.chunk(len(self.async_rollout_workers))
        kv_metas = ray.get(
            [
                worker.generate_sequences.remote(kv_meta_chunk)
                for worker, kv_meta_chunk in zip(self.async_rollout_workers, kv_meta_chunks, strict=True)
            ]
        )
        kv_meta = KVBatchMeta.concat(kv_metas)
        logger.info(f"KVBatchMeta: {kv_meta}")

        return kv_meta


@dataclass
class TrainerConfig:
    """Top-level configuration for :class:`Trainer`."""

    global_batch_size: int = 8
    rollout_agent_num_workers: int = 1
    vocab_size: int = 32000
    agent_loop: AgentLoopConfig = field(default_factory=AgentLoopConfig)
    dataset: MessageDatasetConfig = field(default_factory=MessageDatasetConfig)

    def __post_init__(self):
        self.agent_loop.vocab_size = self.vocab_size
        self.dataset.vocab_size = self.vocab_size


class Trainer:
    def __init__(self, config: TrainerConfig, tq_config):
        self.config = config
        tq.init(tq_config)
        self.tq_client = tq.get_client()
        self.actor_rollout_wg = ActorRolloutRefWorker()
        self.async_rollout_manager = AgentLoopManager(
            num_workers=config.rollout_agent_num_workers,
            agent_loop_config=config.agent_loop,
            tq_config=tq_config,
        )
        self.dataset = MessageDataset(config.dataset)

    def fit(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.global_batch_size,
            shuffle=True,
            collate_fn=message_collate_fn,
        )

        for step, batch in enumerate(dataloader):
            logger.info(f"Step {step}: batch_size = {batch.batch_size[0]}")

            # ========================= Generate keys and put messages to TQ =========================
            batch_keys = [str(uuid.uuid4()) for _ in range(batch.batch_size[0])]
            tq.kv_batch_put(keys=batch_keys, partition_id=f"train_{step}", fields=batch)
            logger.info("demo put messages ok!")
            time.sleep(5)

            # ========================= Sample generate KVBatchMeta =========================
            sampled_keys = random.sample(batch_keys, min(self.config.global_batch_size, len(batch_keys)))
            meta = KVBatchMeta(
                keys=sampled_keys,
                tags=[{} for _ in sampled_keys],
                partition_id=f"train_{step}",
                fields=["messages"],
            )
            logger.info(f"demo get KVBatchMeta {meta}")

            # ========================= Rollout: generate sequences =========================
            meta = self.async_rollout_manager.generate_sequences(meta)
            logger.info(f"demo get after gen KVBatchMeta {meta}")

            # ========================= Compute ref log prob =========================
            meta.fields = ["prompt_ids", "response_ids", "input_ids"]
            meta = self.actor_rollout_wg.compute_ref_log_prob(meta)
            logger.info(f"demo get ref log prob KVBatchMeta: {meta}")

            # ========================= Compute old log prob =========================
            meta.fields = ["prompt_ids", "response_ids", "input_ids"]
            meta = self.actor_rollout_wg.compute_log_prob(meta)
            logger.info(f"demo get old log prob KVBatchMeta: {meta}")

            # ========================= Compute reward =========================
            meta.fields = ["response_ids", "ref_log_prob", "old_log_prob"]
            reward_data = tq.kv_batch_get_by_meta(meta=meta)
            reward_output = compute_reward(reward_data["response_ids"])
            meta = tq.kv_batch_put(keys=meta.keys, partition_id=meta.partition_id, fields=reward_output)
            logger.info(f"demo reward KVBatchMeta: {meta}")

            # ========================= Update actor =========================
            meta.fields = [
                "input_ids",
                "response_ids",
                "response_mask",
                "advantage",
                "old_log_prob",
                "ref_log_prob",
            ]
            meta = self.actor_rollout_wg.update_actor(meta)
            logger.info(f"demo get after update actor KVBatchMeta: {meta}")

            # ========================= Sync weights to rollout =========================
            asyncio.run(self.actor_rollout_wg.update_weights(global_steps=step))
            logger.info("demo update weights done")

            # ========================= Clear partition in TQ =========================
            self.tq_client.clear_partition(partition_id=f"train_{step}")
            logger.info("clear ok!")

        logger.info("demo done!")
        self.tq_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-controller TransferQueue demo")

    # TrainerConfig
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--rollout-agent-num-workers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=32000)

    # AgentLoopConfig
    parser.add_argument("--max-turns-range", type=int, nargs=2, default=[1, 4], metavar=("MIN", "MAX"))
    parser.add_argument("--tool-response-length-range", type=int, nargs=2, default=[5, 20], metavar=("MIN", "MAX"))
    parser.add_argument("--response-length", type=int, default=32)
    parser.add_argument("--image-token-length", type=int, default=64)

    # MessageDatasetConfig
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--text-length-range", type=int, nargs=2, default=[10, 128], metavar=("MIN", "MAX"))
    parser.add_argument("--num-images-range", type=int, nargs=2, default=[0, 3], metavar=("MIN", "MAX"))

    # TQ backend
    parser.add_argument("--num-data-storage-units", type=int, default=2)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainerConfig:
    return TrainerConfig(
        global_batch_size=args.global_batch_size,
        rollout_agent_num_workers=args.rollout_agent_num_workers,
        vocab_size=args.vocab_size,
        agent_loop=AgentLoopConfig(
            max_turns_range=tuple(args.max_turns_range),
            tool_response_length_range=tuple(args.tool_response_length_range),
            response_length=args.response_length,
            image_token_length=args.image_token_length,
        ),
        dataset=MessageDatasetConfig(
            num_samples=args.num_samples,
            text_length_range=tuple(args.text_length_range),
            num_images_range=tuple(args.num_images_range),
        ),
    )


if __name__ == "__main__":
    args = parse_args()
    ray.init()

    trainer_config = build_config(args)

    tq_conf = OmegaConf.load(resources.files("transfer_queue") / "config.yaml")
    tq_conf = OmegaConf.merge(
        tq_conf, {"backend": {"SimpleStorage": {"num_data_storage_units": args.num_data_storage_units}}}
    )

    trainer = Trainer(trainer_config, tq_conf)
    trainer.fit()

    ray.shutdown()
