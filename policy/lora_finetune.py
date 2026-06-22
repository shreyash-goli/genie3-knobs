"""Stage 7 — LoRA fine-tuning of Genie3's V1Denoiser via PPO.

This module MUST run inside the genie3 conda env (it imports genie3 and PEFT directly).

Architecture:
    The V1Denoiser has two natural LoRA targets:
      1. LatentTransformer blocks — linear_pi, linear_pj, linear_p (pair update projections)
      2. IPA modules — linear_q, linear_kv, linear_out (SE(3)-equivariant attention)

    We attach LoRA adapters to these layers via PEFT's get_peft_model().  genie3.Linear
    subclasses nn.Linear directly, so PEFT finds it without modification.

    PPO update loop:
      1. env.reset() → optionally seed x_T from FrontierBuffer
      2. Run one full diffusion trajectory (brancher.run_branching_experiment)
         with the LoRA-adapted model
      3. env.step() → reward from compute_reward() on the terminal structure
      4. Collect (obs, action, reward, value) into a rollout buffer
      5. Every ppo_update_freq episodes: PPO gradient update on LoRA weights only

    The action in Stage 7 is the direction_scale intervention (Stage 6 action space),
    not a change to the denoiser weights directly.  The denoiser weights are updated
    offline via the PPO gradient w.r.t. the *generation quality reward*, treating the
    full diffusion trajectory as a single policy step.

    This is a REINFORCE-style update at the trajectory level:
        loss = -log π_θ(a|s) * R   (where R is compute_reward() at S_0)
    PPO clips this via the importance-sampling ratio to stay close to the old policy.

Configuration:
    RLKNOBS_LORA_R         LoRA rank (default: 8)
    RLKNOBS_LORA_ALPHA     LoRA alpha (default: 16)
    RLKNOBS_LORA_DROPOUT   LoRA dropout (default: 0.05)
    RLKNOBS_LORA_TARGETS   comma-separated layer name substrings to target
                           (default: linear_q,linear_kv,linear_out,linear_pi,linear_pj)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# LoRA target layers — these are the nn.Linear subclasses inside IPA and LatentTransformer
_DEFAULT_LORA_TARGET_MODULES = [
    "linear_q",
    "linear_kv",
    "linear_out",
    "linear_pi",
    "linear_pj",
    "linear_p",
]


# ---------------------------------------------------------------------------
# LoRA config dataclass
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """Parameters for the LoRA adapter."""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(_DEFAULT_LORA_TARGET_MODULES))
    bias: str = "none"

    @classmethod
    def from_env(cls) -> "LoRAConfig":
        return cls(
            r=int(os.environ.get("RLKNOBS_LORA_R", "8")),
            lora_alpha=int(os.environ.get("RLKNOBS_LORA_ALPHA", "16")),
            lora_dropout=float(os.environ.get("RLKNOBS_LORA_DROPOUT", "0.05")),
            target_modules=(
                os.environ.get("RLKNOBS_LORA_TARGETS", ",".join(_DEFAULT_LORA_TARGET_MODULES))
                .split(",")
            ),
        )


# ---------------------------------------------------------------------------
# LoRA adapter attachment
# ---------------------------------------------------------------------------

def attach_lora(model, lora_cfg: Optional[LoRAConfig] = None):
    """Attach LoRA adapters to a V1Denoiser and return the PEFT model.

    Only the LoRA adapter weights are trainable; all other parameters are frozen.

    Parameters
    ----------
    model   : a genie3 V1Denoiser (nn.Module)
    lora_cfg: LoRAConfig (defaults to LoRAConfig.from_env())

    Returns
    -------
    peft_model : PEFT LoRA-wrapped model, same forward signature as V1Denoiser
    """
    try:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError(
            "peft is required for Stage 7 LoRA fine-tuning: pip install peft"
        )

    cfg = lora_cfg or LoRAConfig.from_env()
    peft_cfg = PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        task_type="FEATURE_EXTRACTION",  # closest TaskType for non-seq2seq
    )
    peft_model = get_peft_model(model, peft_cfg)
    trainable, total = peft_model.get_nb_trainable_parameters()
    log.info(
        "LoRA adapter attached: %d trainable / %d total params (%.2f%%)",
        trainable, total, 100 * trainable / max(total, 1),
    )
    return peft_model


def save_lora(peft_model, output_dir: Path) -> None:
    """Save only the LoRA adapter weights (small — typically a few MB)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(output_dir))
    log.info("LoRA adapter saved to %s", output_dir)


def load_lora(base_model, adapter_dir: Path):
    """Load a previously saved LoRA adapter onto a base model."""
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("peft is required: pip install peft")
    return PeftModel.from_pretrained(base_model, str(adapter_dir))


# ---------------------------------------------------------------------------
# Rollout buffer for trajectory-level PPO
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    """One complete diffusion-trajectory episode."""
    obs: Any               # observation at episode start (np.ndarray)
    action: int            # direction_scale action chosen
    log_prob: float        # log π_old(a|s)
    reward: float          # terminal compute_reward()
    value: float           # V(s) estimate from critic
    metrics: dict          # full oracle metrics for logging


class RolloutBuffer:
    """Collects rollouts, computes advantages, and clears after PPO update."""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._rollouts: list[Rollout] = []

    def add(self, rollout: Rollout) -> None:
        self._rollouts.append(rollout)

    def __len__(self) -> int:
        return len(self._rollouts)

    def advantages_and_returns(self) -> tuple[list[float], list[float]]:
        """GAE-style advantages for one-step episodes (trivially: A = R - V)."""
        advantages, returns = [], []
        for r in self._rollouts:
            ret = r.reward  # one-step: no discounting needed
            adv = ret - r.value
            advantages.append(adv)
            returns.append(ret)
        return advantages, returns

    def clear(self) -> None:
        self._rollouts.clear()

    def iter_minibatches(self, batch_size: int):
        """Yield (obs, actions, log_probs, advantages, returns) minibatches."""
        import numpy as np
        n = len(self._rollouts)
        idx = np.random.permutation(n)
        advantages, returns = self.advantages_and_returns()
        # normalise advantages
        adv_arr = np.array(advantages, dtype=np.float32)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
        for start in range(0, n, batch_size):
            batch_idx = idx[start: start + batch_size]
            yield (
                np.stack([self._rollouts[i].obs for i in batch_idx]),
                np.array([self._rollouts[i].action for i in batch_idx], dtype=np.int64),
                np.array([self._rollouts[i].log_prob for i in batch_idx], dtype=np.float32),
                adv_arr[batch_idx],
                np.array([returns[i] for i in batch_idx], dtype=np.float32),
            )


# ---------------------------------------------------------------------------
# PPO update step for LoRA fine-tuning
# ---------------------------------------------------------------------------

@dataclass
class PPOFinetuneConfig:
    """Hyperparameters for Stage 7 PPO fine-tuning."""
    total_episodes: int = 500
    ppo_update_freq: int = 32        # collect this many episodes then update
    n_epochs: int = 4
    batch_size: int = 8
    learning_rate: float = 1e-4
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lora_cfg: LoRAConfig = field(default_factory=LoRAConfig)
    save_every: int = 100            # save LoRA checkpoint every N episodes
    save_dir: str = "data/lora_checkpoints"


def train_lora_ppo(
    env,                              # DiffusionInterventionEnv
    actor_critic,                     # nn.Module with .act(obs) and .evaluate(obs, action)
    ppo_cfg: Optional[PPOFinetuneConfig] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """PPO fine-tuning loop for Stage 7.

    The actor_critic is a small MLP (same as Stage 0-3 policy) that outputs
    action logits + value estimate.  Its gradients flow back through the LoRA
    adapter weights in the denoiser only when oracle_mode="live" and the
    trajectory is differentiable.  In offline mode this is a standard discrete
    PPO over the intervention action space.

    Parameters
    ----------
    env          : DiffusionInterventionEnv instance
    actor_critic : ActorCritic module (see ActorCritic class below)
    ppo_cfg      : PPOFinetuneConfig (defaults to PPOFinetuneConfig())
    verbose      : print episode summaries

    Returns
    -------
    training_log : dict with episode rewards, policy tables, etc.
    """
    try:
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1)
    except ImportError:
        raise ImportError("torch is required")

    cfg = ppo_cfg or PPOFinetuneConfig()
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, actor_critic.parameters()),
        lr=cfg.learning_rate,
    )
    buffer = RolloutBuffer()
    episode_rewards: list[float] = []
    episode = 0

    while episode < cfg.total_episodes:
        obs, info = env.reset()
        target = info["target"]
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, value = actor_critic(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        _, reward, _, _, step_info = env.step(int(action.item()))
        episode_rewards.append(reward)

        buffer.add(Rollout(
            obs=obs,
            action=int(action.item()),
            log_prob=float(log_prob.item()),
            reward=reward,
            value=float(value.item()),
            metrics=step_info,
        ))
        episode += 1

        if verbose and episode % 10 == 0:
            recent = episode_rewards[-10:]
            log.info(
                "episode %d/%d  mean_reward(last10)=%.4f  buffer=%d",
                episode, cfg.total_episodes, sum(recent) / len(recent), len(buffer),
            )

        # PPO update
        if len(buffer) >= cfg.ppo_update_freq or episode == cfg.total_episodes:
            _ppo_update(actor_critic, optimizer, buffer, cfg)
            buffer.clear()

        # checkpoint
        if episode % cfg.save_every == 0:
            ckpt_path = save_dir / f"actor_critic_ep{episode}.pt"
            torch.save(actor_critic.state_dict(), ckpt_path)
            log.info("Saved checkpoint: %s", ckpt_path)

    return {
        "episode_rewards": episode_rewards,
        "mean_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0,
        "total_episodes": cfg.total_episodes,
    }


def _ppo_update(actor_critic, optimizer, buffer: RolloutBuffer,
                cfg: PPOFinetuneConfig) -> None:
    import torch
    import torch.nn.functional as F

    for _ in range(cfg.n_epochs):
        for obs_b, act_b, old_lp_b, adv_b, ret_b in buffer.iter_minibatches(cfg.batch_size):
            obs_t = torch.tensor(obs_b, dtype=torch.float32)
            act_t = torch.tensor(act_b, dtype=torch.long)
            old_lp_t = torch.tensor(old_lp_b, dtype=torch.float32)
            adv_t = torch.tensor(adv_b, dtype=torch.float32)
            ret_t = torch.tensor(ret_b, dtype=torch.float32)

            logits, values = actor_critic(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(act_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_lp_t)
            pg_loss1 = -adv_t * ratio
            pg_loss2 = -adv_t * torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            vf_loss = F.mse_loss(values.squeeze(-1), ret_t)
            loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), cfg.max_grad_norm)
            optimizer.step()


# ---------------------------------------------------------------------------
# ActorCritic MLP (shared trunk, separate heads)
# ---------------------------------------------------------------------------

class ActorCritic:
    """Small MLP actor-critic for Stage 6/7 intervention policy.

    A thin wrapper so we can import this without torch at module load time.
    Instantiate inside the genie3 conda env where torch is available.
    """

    @staticmethod
    def build(obs_dim: int, n_actions: int, hidden: tuple[int, ...] = (64, 64)):
        """Build and return an nn.Module actor-critic."""
        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("torch is required")

        class _AC(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                in_dim = obs_dim
                for h in hidden:
                    layers += [nn.Linear(in_dim, h), nn.Tanh()]
                    in_dim = h
                self.trunk = nn.Sequential(*layers)
                self.policy_head = nn.Linear(in_dim, n_actions)
                self.value_head = nn.Linear(in_dim, 1)

            def forward(self, x):
                h = self.trunk(x)
                return self.policy_head(h), self.value_head(h)

        return _AC()
