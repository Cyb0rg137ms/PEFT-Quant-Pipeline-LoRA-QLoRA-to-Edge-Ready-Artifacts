"""Group Relative Policy Optimisation (GRPO, Shao et al. 2024) with LoRA adapters, on a toy task.

For each prompt, ``G`` completions are sampled; each gets a scalar reward and
its advantage is the group-normalised reward ``(r - mean) / (std + eps)``
(no learned value function). The loss is ``-mean(A * log pi(completion))``
over the sampled tokens, plus an optional KL penalty to the frozen reference
(the base model with the adapter switched off, computed exactly here since
the toy policy is factorised).

The toy environment: the prompt is ``L`` random tokens from a vocabulary of
``V``; the reward is the fraction of positions where the completion equals the
prompt reversed. The policy is a frozen random MLP with LoRA on its hidden
layers, so only adapter parameters learn. This demonstrates the algorithm
end-to-end on CPU in seconds; it is *not* an LLM fine-tuning result.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from peftquant.inject import adapters_disabled, count_parameters, inject_adapters


@dataclass
class GRPOConfig:
    vocab: int = 8
    length: int = 6
    hidden: int = 128
    group_size: int = 8
    prompts_per_step: int = 32
    steps: int = 400
    lr: float = 3e-3
    rank: int = 8
    kl_coef: float = 0.0
    clip_grad: float = 1.0
    seed: int = 0


class ToyPolicy(nn.Module):
    """Prompt tokens -> independent categorical per output position."""

    def __init__(self, cfg: GRPOConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, 16)
        self.net = nn.Sequential(
            nn.Linear(cfg.length * 16, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.length * cfg.vocab),
        )

    def forward(self, prompt: Tensor) -> Tensor:
        h = self.embed(prompt).flatten(1)
        return self.net(h).view(-1, self.cfg.length, self.cfg.vocab)


def reward_fn(prompt: Tensor, completion: Tensor) -> Tensor:
    """Fraction of positions matching the reversed prompt; shape ``(batch,)``."""
    return (completion == prompt.flip(1)).float().mean(1)


def group_advantages(rewards: Tensor, eps: float = 1e-4) -> Tensor:
    """``rewards`` shape ``(prompts, G)`` -> per-sample group-normalised advantages."""
    mean = rewards.mean(1, keepdim=True)
    std = rewards.std(1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


@torch.no_grad()
def evaluate(policy: ToyPolicy, n: int = 512, seed: int = 123) -> float:
    """Mean reward of the greedy policy on fresh prompts."""
    g = torch.Generator().manual_seed(seed)
    prompt = torch.randint(0, policy.cfg.vocab, (n, policy.cfg.length), generator=g)
    return float(reward_fn(prompt, policy(prompt).argmax(-1)).mean())


def train_grpo(cfg: GRPOConfig = GRPOConfig()) -> dict[str, float | list[float]]:
    """Train LoRA adapters with GRPO; returns measured before/after rewards and the curve."""
    torch.manual_seed(cfg.seed)
    policy = ToyPolicy(cfg)
    inject_adapters(policy, r"net\.[02]", kind="lora", rank=cfg.rank, alpha=2.0 * cfg.rank)
    trainable, total = count_parameters(policy)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=cfg.lr)
    before = evaluate(policy)
    curve: list[float] = []
    for _ in range(cfg.steps):
        prompt = torch.randint(0, cfg.vocab, (cfg.prompts_per_step, cfg.length))
        logits = policy(prompt)  # (P, L, V)
        logp = F.log_softmax(logits, -1)
        dist = torch.distributions.Categorical(logits=logits.unsqueeze(1).expand(-1, cfg.group_size, -1, -1))
        comp = dist.sample()  # (P, G, L)
        rewards = reward_fn(prompt.repeat_interleave(cfg.group_size, 0), comp.flatten(0, 1)).view(-1, cfg.group_size)
        adv = group_advantages(rewards)
        tok_logp = logp.unsqueeze(1).expand(-1, cfg.group_size, -1, -1).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
        loss = -(adv.unsqueeze(-1) * tok_logp).mean()
        if cfg.kl_coef > 0:
            with adapters_disabled(policy), torch.no_grad():
                ref = F.log_softmax(policy(prompt), -1)
            loss = loss + cfg.kl_coef * (logp.exp() * (logp - ref)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.clip_grad)
        opt.step()
        curve.append(float(rewards.mean()))
    return {
        "greedy_reward_before": before,
        "greedy_reward_after": evaluate(policy),
        "chance_reward": 1.0 / cfg.vocab,
        "trainable_params": trainable,
        "total_params": total,
        "sampled_reward_curve": curve,
    }
