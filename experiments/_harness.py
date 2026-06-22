"""Shared evaluation harness for the fixed / random / bandit / PPO comparison.

Keeps the per-stage scripts thin.  Responsibilities:

* train the online learners (bandit, PPO) under a *matched* env-interaction budget;
* evaluate every policy under an identical protocol (deterministic/greedy action, many
  oracle samples per target to average over the offline oracle's sampling noise);
* repeat across seeds for confidence intervals;
* log per-episode rows (JSONL) and a run summary (JSON) under data/experiment_logs/.

A "policy" is anything with ``select(obs, target) -> int`` and ``update(target, action,
reward) -> None`` (fixed/random/bandit/PPO-wrapper all satisfy this).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

import config


def _jsonable(o):
    """default= hook so numpy scalars/arrays (e.g. int64 action indices) serialise."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
from baselines.contextual_bandit import ContextualBandit
from baselines.fixed_heuristic import FixedHeuristic
from baselines.random_policy import RandomPolicy
from envs.genie_branch_env import GenieBranchEnv
from oracle.reward_oracle import OfflineRewardModel
from policy.train_ppo import PPOPolicyWrapper, train_ppo


# --------------------------------------------------------------------------------------
def train_online(policy, env, budget: int, seed: int = 0) -> None:
    """Train an online learner (e.g. bandit) for ``budget`` episodes, round-robin targets."""
    env.reset_round()
    obs, info = env.reset(seed=seed)
    for _ in range(budget):
        target = info["target"]
        action = policy.select(obs, target)
        obs, reward, terminated, truncated, info = env.step(action)
        policy.update(target, action, float(reward))
        obs, info = env.reset()  # one-step episodes


def evaluate_policy(policy, env, n_per_target: int = 200,
                    seed: int = 0) -> dict[str, Any]:
    """Evaluate a (frozen) policy: for every target, sample ``n_per_target`` episodes and
    average reward / success / chosen-action distribution over the oracle's noise."""
    rng_env = GenieBranchEnv  # type hint only
    env.reset_round()
    per_target: dict[str, dict] = {}
    all_rewards: list[float] = []
    rows: list[dict] = []
    for target in env.targets:
        rewards, successes, actions, backoffs = [], [], [], []
        for k in range(n_per_target):
            obs, info = env.reset(seed=seed + k, options={"target": target})
            action = policy.select(obs, target)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            successes.append(1.0 if info.get("complex_success") else 0.0)
            actions.append(int(action))
            backoffs.append(info.get("backoff"))
            rows.append({
                "policy": getattr(policy, "name", policy.__class__.__name__),
                "target": target, "action": int(action), "reward": float(reward),
                "complex_success": bool(info.get("complex_success")),
                "backoff": info.get("backoff"),
                "levers": info.get("action"),
            })
        per_target[target] = {
            "mean_reward": float(np.mean(rewards)),
            "success_rate": float(np.mean(successes)),
            "modal_action": int(np.bincount(actions).argmax()),
            "action_levers": env.decode_action(int(np.bincount(actions).argmax())),
        }
        all_rewards.extend(rewards)
    return {
        "overall_mean_reward": float(np.mean(all_rewards)),
        "per_target": per_target,
        "rows": rows,
    }


@dataclass
class StageConfig:
    name: str
    levers: tuple
    targets: Optional[list] = None
    seeds: tuple = (0, 1, 2)
    train_budget: int = 4000      # online-learner env interactions (bandit)
    ppo_timesteps: int = 20_000
    eval_per_target: int = 200
    bandit_warm_start: bool = False
    source_sweep: Optional[str] = None


def _make_env(stage: StageConfig, seed: int) -> GenieBranchEnv:
    oracle = OfflineRewardModel(source_sweep=stage.source_sweep)
    return GenieBranchEnv(
        oracle=oracle,
        targets=stage.targets,
        levers=stage.levers,
        seed=seed,
    )


def run_stage(stage: StageConfig, verbose: bool = True) -> dict[str, Any]:
    """Run the four-way comparison for a stage and write logs.  Returns the summary dict."""
    config.ensure_data_dirs()
    t0 = time.time()
    log_dir = config.EXPERIMENTS_LOG_DIR / stage.name
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_log = (log_dir / "episodes.jsonl").open("w")

    # collect per-seed overall means for each policy
    results: dict[str, list[float]] = {"fixed": [], "random": [], "bandit": [], "ppo": []}
    # keep the last seed's per-target detail + bandit/ppo policy tables for inspection
    detail: dict[str, Any] = {}

    for seed in stage.seeds:
        env = _make_env(stage, seed)
        n_actions = env.action_space.n
        if verbose:
            print(f"[{stage.name}] seed={seed}  |A|={n_actions}  targets={len(env.targets)}")

        # --- fixed heuristic ---------------------------------------------------------
        fixed = FixedHeuristic(env)
        ev = evaluate_policy(fixed, env, stage.eval_per_target, seed=seed)
        results["fixed"].append(ev["overall_mean_reward"])
        _dump_rows(episode_log, ev["rows"], stage.name, seed)
        detail["fixed"] = ev["per_target"]
        detail["fixed_action"] = env.decode_action(fixed.best_action)

        # --- random ------------------------------------------------------------------
        rand = RandomPolicy(n_actions, seed=seed)
        ev = evaluate_policy(rand, env, stage.eval_per_target, seed=seed)
        results["random"].append(ev["overall_mean_reward"])
        _dump_rows(episode_log, ev["rows"], stage.name, seed)
        detail["random"] = ev["per_target"]

        # --- contextual bandit (trained online under the budget) ---------------------
        bandit = ContextualBandit(env, exploration="ucb",
                                  warm_start=stage.bandit_warm_start, seed=seed)
        train_online(bandit, env, stage.train_budget, seed=seed)
        ev = evaluate_policy(bandit, env, stage.eval_per_target, seed=seed)
        results["bandit"].append(ev["overall_mean_reward"])
        _dump_rows(episode_log, ev["rows"], stage.name, seed)
        detail["bandit"] = ev["per_target"]
        detail["bandit_policy_table"] = {t: env.decode_action(a)
                                         for t, a in bandit.policy_table().items()}

        # --- PPO ---------------------------------------------------------------------
        ppo_env = _make_env(stage, seed)
        model = train_ppo(ppo_env, total_timesteps=stage.ppo_timesteps, seed=seed)
        ppo = PPOPolicyWrapper(model, ppo_env)
        ev = evaluate_policy(ppo, env, stage.eval_per_target, seed=seed)
        results["ppo"].append(ev["overall_mean_reward"])
        _dump_rows(episode_log, ev["rows"], stage.name, seed)
        detail["ppo"] = ev["per_target"]
        detail["ppo_policy_table"] = _ppo_policy_table(ppo, env)

    episode_log.close()

    summary = {
        "stage": stage.name,
        "levers": list(stage.levers),
        "targets": list((stage.targets or _make_env(stage, 0).targets)),
        "n_actions": _make_env(stage, 0).action_space.n,
        "seeds": list(stage.seeds),
        "train_budget": stage.train_budget,
        "ppo_timesteps": stage.ppo_timesteps,
        "eval_per_target": stage.eval_per_target,
        "bandit_warm_start": stage.bandit_warm_start,
        "scores": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                       "per_seed": v} for k, v in results.items()},
        "ppo_beats_bandit": float(np.mean(results["ppo"])) > float(np.mean(results["bandit"])),
        "ppo_minus_bandit": float(np.mean(results["ppo"]) - np.mean(results["bandit"])),
        "detail_last_seed": detail,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    if verbose:
        _print_summary(summary)
    return summary


def _ppo_policy_table(ppo: PPOPolicyWrapper, env) -> dict[str, Any]:
    table = {}
    for t in env.targets:
        obs, _ = env.reset(options={"target": t})
        a = ppo.select(obs, t)
        table[t] = env.decode_action(a)
    return table


def _dump_rows(fh, rows, stage_name, seed) -> None:
    for r in rows:
        r = dict(r, stage=stage_name, seed=seed)
        fh.write(json.dumps(r, default=_jsonable) + "\n")


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\n=== {summary['stage']} :: |A|={summary['n_actions']} "
          f"levers={summary['levers']} ===")
    for k in ("fixed", "random", "bandit", "ppo"):
        s = summary["scores"][k]
        print(f"  {k:7s}  mean_reward = {s['mean']:.4f} ± {s['std']:.4f}")
    verdict = "PPO BEATS bandit" if summary["ppo_beats_bandit"] else "PPO does NOT beat bandit"
    print(f"  -> {verdict}  (Δ = {summary['ppo_minus_bandit']:+.4f})")
    print(f"  wall: {summary['wall_seconds']}s   logs: data/experiment_logs/{summary['stage']}/")
