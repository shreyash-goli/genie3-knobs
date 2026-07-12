"""FK (Feynman-Kac) Steering as an inference-time layer (NEXT_STEPS.md §7.2).

FK Steering runs *k* parallel trajectories, scores them at a few score-steps with a potential
function, resamples (kill low-scoring particles, duplicate high-scoring ones), and takes the
highest-reward survivor at the end. This wraps a trained policy at *inference* only -- PPO
training is unchanged (§7.2), preserving its on-policy assumptions.

Scope of this module (the offline first cut, per the approved plan):
    * Particles are `k` independent, distinctly-seeded `DiffusionInterventionEnv` instances
      rolled out in lockstep. Offline, they share conditioning but differ in the oracle's
      stochastic child draw (`OfflineRewardModel.sample` -> `rng.choice(pool)`), which is the
      "different noise realizations under the same conditioning" that FK selects among.
    * Particles are scored at t=0 (the terminal step) with the true terminal reward
      (`potential_fn` defaults to `compute_reward`). §7.1 already found the zero-cost
      mid-trajectory geometric potentials (rg, nc_termini) too weak to steer on, and true
      mid-trajectory kill/duplicate needs cloning a partial diffusion state -- a live/genie3
      capability, not available offline. So the default path is best-of-k with the resample
      primitive available (and a `potential_fn`/`score_steps` hook left open) for that later
      live phase.

Two policy-integration variants (§7.2):
    * shared_action (default, recommended): the policy emits ONE action per step (argmax over
      the mean particle observation); all k particles apply it. FK then selects among noise
      realizations only -- you always know the surviving trajectory used the policy's
      conditioning choice.
    * independent_actions: each particle samples its own action from Categorical(logits); FK
      selects over conditioning *and* noise jointly (more powerful, less interpretable).

Pass `actor_critic=None` to condition randomly (the §7.3 "random conditioning" baselines).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from envs.commitment_window import HOTSPOT_MODES
from oracle.reward_oracle import compute_reward


def fk_resample(
    scores: np.ndarray, rng: np.random.Generator, temperature: float = 0.0
) -> np.ndarray:
    """FK resampling: kill low-scoring particles, duplicate high-scoring ones.

    Returns an index array of length `len(scores)` -- the surviving particle indices
    (with duplicates), sampled multinomially from the softmax of `scores / temperature`.

    * temperature <= 0 : deterministic -- every survivor is the argmax particle (the
      degenerate best-of-k case; the closest existing primitive is `_aggregate_children`'s
      best-child-by-iptm argmax in live_oracle.py).
    * temperature -> inf : survivors sampled ~uniformly (no selection pressure).
    * len(scores) <= 1  : identity.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n <= 1:
        return np.arange(n)
    if temperature <= 0:
        return np.full(n, int(np.argmax(scores)), dtype=np.int64)
    z = scores / temperature
    z = z - z.max()  # numerically stable softmax
    w = np.exp(z)
    w /= w.sum()
    return rng.choice(n, size=n, p=w)


def _policy_argmax(actor_critic, obs: np.ndarray) -> int:
    import torch
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _ = actor_critic(obs_t)
    return int(logits.argmax(dim=-1).item())


def _policy_sample(actor_critic, obs: np.ndarray, rng: np.random.Generator) -> int:
    import torch
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _ = actor_critic(obs_t)
    # seed torch's sampler from the numpy rng so the whole rollout is reproducible
    gen = torch.Generator().manual_seed(int(rng.integers(2**31)))
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=gen).item())


def fk_rollout(
    make_env_seeded: Callable[[int], Any],
    actor_critic: Optional[Any],
    k: int,
    rng: np.random.Generator,
    *,
    base_seed: int,
    independent_actions: bool = False,
    temperature: float = 0.0,
    potential_fn: Optional[Callable[[dict], float]] = None,
    return_diagnostics: bool = False,
):
    """Run one FK-steered episode with `k` particles; return the best survivor's terminal reward.

    Parameters
    ----------
    make_env_seeded : factory `seed -> DiffusionInterventionEnv`. Build the env with
        `intermediate_reward_scale=0.0` so a particle's terminal step reward *is* its terminal
        `compute_reward` (and k=1 reduces exactly to a single greedy/random episode).
    actor_critic : the trained policy, or None for random conditioning (§7.3 baselines).
    k : number of particles. k=1 is the no-FK single-trajectory case (a built-in sanity check:
        it must equal a plain `_eval_policy` episode on the same env).
    independent_actions : False = shared-action variant (recommended), True = per-particle
        sampling variant (§7.2).
    temperature : FK resample temperature. 0.0 (default) = best-of-k; >0 = softer selection.
    potential_fn : particle scorer, `metrics_dict -> float`. Defaults to `compute_reward`. Left
        pluggable for a future mid-trajectory / learned potential (§7.1's deferred direction).

    Returns
    -------
    float  (best surviving terminal reward), or, if `return_diagnostics`,
    (best_reward, {"terminal_rewards", "survivors", "target", "length_delta", "n_steps"}).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    score = potential_fn or compute_reward

    # Sample the shared episode context once, so all particles condition on the same target and
    # binder length (only the oracle's noise draw differs across particles).
    probe = make_env_seeded(base_seed)
    from envs.commitment_window import LENGTH_DELTAS
    target = str(rng.choice(probe.targets))
    length_delta = int(rng.choice(LENGTH_DELTAS))

    particles = [make_env_seeded(base_seed + i) for i in range(k)]
    obs = [
        env.reset(seed=base_seed + i,
                  options={"target": target, "length_delta": length_delta})[0]
        for i, env in enumerate(particles)
    ]
    done = [False] * k
    terminal_reward = [0.0] * k
    n_steps = 0

    while not all(done):
        active = [i for i in range(k) if not done[i]]
        if independent_actions:
            actions = {i: _select_action(actor_critic, obs[i], rng, sample=True)
                       for i in active}
        else:
            # shared action: one choice from the mean observation over active particles
            mean_obs = np.mean([obs[i] for i in active], axis=0)
            a = _select_action(actor_critic, mean_obs, rng, sample=False)
            actions = {i: a for i in active}

        for i in active:
            o, r, term, trunc, info = particles[i].step(actions[i])
            obs[i] = o
            if term or trunc:
                done[i] = True
                terminal_reward[i] = float(score(info))
        n_steps += 1

    # FK resample at the (terminal) score-step, then take the highest-reward survivor (§7).
    scores = np.asarray(terminal_reward, dtype=np.float64)
    survivors = fk_resample(scores, rng, temperature=temperature)
    best = float(scores[survivors].max())

    if return_diagnostics:
        return best, {
            "terminal_rewards": terminal_reward,
            "survivors": survivors.tolist(),
            "target": target,
            "length_delta": length_delta,
            "n_steps": n_steps,
        }
    return best


def _select_action(
    actor_critic: Optional[Any], obs: np.ndarray, rng: np.random.Generator, *, sample: bool
) -> int:
    """One action for a particle. actor_critic=None -> random hotspot mode (never commit, so
    the random baseline behaves like `_random_policy` in ppo_vs_bandit_offline.py)."""
    if actor_critic is None:
        return int(rng.integers(len(HOTSPOT_MODES)))
    if sample:
        return _policy_sample(actor_critic, obs, rng)
    return _policy_argmax(actor_critic, obs)
