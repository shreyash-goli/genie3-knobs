# LatProtRL — reference only

This directory is a placeholder for the **LatProtRL** repo (Lee et al., 2024, *Robust
Optimization in Protein Fitness Landscapes Using RL in Latent Space*), kept as **read-only
reference** for porting:

- the MDP formulation + PPO loop (Algorithm 2), and
- the **Frontier Buffer** (Algorithm 3) — `INITIALIZE` / `TOP` / `UPDATE`.

The Frontier Buffer signatures are mirrored (stub only, not wired) in
[`buffer/frontier_buffer.py`](../../buffer/frontier_buffer.py).

Do **not** clone-and-mutate LatProtRL internals here; this project only reads its algorithm
descriptions. On a networked machine, place a checkout here for convenient reference:

```bash
git clone <latprotrl-url> external/latprotrl_ref/latprotrl
```
