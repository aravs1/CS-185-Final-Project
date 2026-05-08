from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor          # [N, L]
    attention_mask: torch.Tensor     # [N, L]
    completion_mask: torch.Tensor    # [N, L-1] float
    old_logprobs: torch.Tensor       # [N, L-1]
    ref_logprobs: torch.Tensor       # [N, L-1]
    rewards: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]

    task_names: Optional[list] = None
    completion_texts: Optional[list] = None

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            completion_mask=self.completion_mask.to(device, non_blocking=True),
            old_logprobs=self.old_logprobs.to(device, non_blocking=True),
            ref_logprobs=self.ref_logprobs.to(device, non_blocking=True),
            rewards=self.rewards.to(device, non_blocking=True),
            advantages=self.advantages.to(device, non_blocking=True),
            task_names=self.task_names,
            completion_texts=self.completion_texts,
        )


def iter_minibatches(
    batch: RolloutBatch,
    minibatch_size: int,
    shuffle: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Iterator[RolloutBatch]:

    # TODO(student): iterate over the rollout in minibatches, optionally shuffling the row indices,
    # and yield RolloutBatch objects containing the selected subset.

    N = batch.input_ids.shape[0]

    if shuffle:
        indices = torch.randperm(N, generator=generator)
    else:
        indices = torch.arange(N)

    for start in range(0, N, minibatch_size):
        i = indices[start: start + minibatch_size]

        if batch.task_names:
            task_names = [batch.task_names[x] for x in i.tolist()]
        else:
            task_names = None
        
        if batch.completion_texts:
            completion_texts = [batch.completion_texts[x] for x in i.tolist()]
        else:
            completion_texts = None

        minibatch = RolloutBatch(
            input_ids=batch.input_ids[i],
            attention_mask=batch.attention_mask[i],
            completion_mask=batch.completion_mask[i],
            old_logprobs=batch.old_logprobs[i],
            ref_logprobs=batch.ref_logprobs[i],
            rewards=batch.rewards[i],
            advantages=batch.advantages[i],
            task_names=task_names,
            completion_texts=completion_texts,
        )

        if device is not None:
            minibatch = minibatch.to(device)
        
        yield minibatch
        
