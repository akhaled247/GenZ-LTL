"""Measure whether env_net (80->64) uses walls_lidar or effectively discards it.

Usage:
  python GenZ-LTL/scripts/analyze_walls_env_net.py
  python GenZ-LTL/scripts/analyze_walls_env_net.py --checkpoint path/to/status.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "GenZ-LTL" / "src"))

from model.env.standard_env_net import StandardEnvNet  # noqa: E402

# sar_v1 layout: agent(16) | buildings(16) | walls(16) | reach(16) | avoid(16)
AGENT_DIM = 16
LIDAR_BINS = 16
SLICES = {
    "agent": (0, 16),
    "buildings": (16, 32),
    "walls": (32, 48),
    "reach": (48, 64),
    "avoid": (64, 80),
}


def build_env_net(in_dim: int = 80) -> StandardEnvNet:
    return StandardEnvNet(in_dim, [128, 64], nn.Tanh)


def first_linear(env_net: StandardEnvNet) -> nn.Linear:
    return env_net.mlp[0]


def slice_weight_norms(env_net: StandardEnvNet) -> dict[str, float]:
    w = first_linear(env_net).weight.detach()
    return {name: w[:, lo:hi].norm().item() for name, (lo, hi) in SLICES.items()}


def ablation_sensitivity(
    env_net: StandardEnvNet,
    x: torch.Tensor,
    *,
    perturb_scale: float = 1.0,
) -> dict[str, float]:
    """Relative ||f(x) - f(x with slice zeroed)|| / ||f(x)|| per slice."""
    env_net.eval()
    with torch.no_grad():
        base = env_net(x)
        base_norm = base.norm(dim=-1).clamp(min=1e-8)
        out = {}
        for name, (lo, hi) in SLICES.items():
            x_abl = x.clone()
            x_abl[..., lo:hi] = 0.0
            delta = (env_net(x_abl) - base).norm(dim=-1) / base_norm
            out[name] = delta.mean().item()

        x_wall = x.clone()
        x_wall[..., 32:48] = torch.randn_like(x_wall[..., 32:48]) * perturb_scale
        wall_pert = (env_net(x_wall) - base).norm(dim=-1) / base_norm
        out["walls_random_perturb"] = wall_pert.mean().item()
    return out


def max_walls_influence(env_net: StandardEnvNet, x: torch.Tensor) -> float:
    """Max over walls slice of |d(f_i)/d(x_walls_j)| via autograd (one batch)."""
    env_net.eval()
    x = x.clone().requires_grad_(True)
    out = env_net(x)
    max_grad = 0.0
    for i in range(out.shape[-1]):
        grad = torch.autograd.grad(out[:, i].sum(), x, retain_graph=True)[0]
        g = grad[..., 32:48].abs().max().item()
        max_grad = max(max_grad, g)
    return max_grad


def load_env_net_from_checkpoint(path: Path) -> tuple[StandardEnvNet, dict[str, float]]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("model_state", blob)
    w0 = state["env_net.mlp.0.weight"]
    in_dim = int(w0.shape[1])
    env_net = build_env_net(in_dim)
    env_net.load_state_dict(
        {k.removeprefix("env_net."): v for k, v in state.items() if k.startswith("env_net.")},
        strict=True,
    )
    return env_net, {"in_dim": in_dim, "out_dim": int(state["env_net.mlp.2.weight"].shape[0])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    label = "random_init"
    env_net = build_env_net(80)

    if args.checkpoint is not None:
        env_net, meta = load_env_net_from_checkpoint(args.checkpoint)
        label = f"checkpoint(in={meta['in_dim']})"

    x = torch.randn(args.batches, 80)

    print(f"=== env_net walls analysis ({label}) ===")
    print("Architecture: Linear(80,128) -> Tanh -> Linear(128,64) -> Tanh")
    print()

    norms = slice_weight_norms(env_net)
    total = sum(norms.values())
    print("First-layer weight column norms (||W[:,slice]||_F):")
    for name, n in norms.items():
        pct = 100.0 * n / total if total else 0.0
        print(f"  {name:10s}: {n:.4f}  ({pct:5.1f}% of slice norms sum)")

    # Compare walls vs buildings — if walls << buildings after training, likely unused
    if norms["walls"] > 0:
        ratio = norms["walls"] / norms["buildings"]
        print(f"  walls/buildings norm ratio: {ratio:.4f}")

    print()
    abl = ablation_sensitivity(env_net, x)
    print("Ablation sensitivity (mean ||f(x)-f(x\\slice=0)|| / ||f(x)||):")
    for name, v in abl.items():
        print(f"  {name:20s}: {v:.4f}")

    print()
    g = max_walls_influence(env_net, x[:32])
    print(f"Max |d(env_net)/d(walls)| on random batch: {g:.6f}")

    print()
    # Simulated "complete discard": zero walls columns in first layer
    sim = build_env_net(80)
    if args.checkpoint is not None:
        sim.load_state_dict(env_net.state_dict())
    with torch.no_grad():
        sim.mlp[0].weight[:, 32:48] = 0.0
    sim_abl = ablation_sensitivity(sim, x)
    print("If walls columns were ZEROED in W1 (simulated complete discard):")
    print(f"  walls ablation sensitivity would drop: {abl['walls']:.4f} -> {sim_abl['walls']:.4f}")
    print(f"  walls random perturb response: {abl.get('walls_random_perturb', 0):.4f} -> {sim_abl.get('walls_random_perturb', 0):.4f}")

    print()
    if args.checkpoint is None:
        print("No checkpoint provided — this is UNTRAINED orthogonal init.")
        print("Re-run with --checkpoint pointing at L1-best eval/training status .pth")
    else:
        discarded = abl["walls"] < 0.01 and abl.get("walls_random_perturb", 1) < 0.01
        weak = norms["walls"] / norms["buildings"] < 0.1 if norms["buildings"] else False
        if discarded:
            print("VERDICT: walls_lidar appears DISCARDED (ablation/perturb ~0).")
        elif weak:
            print("VERDICT: walls_lidar WEAKLY used (weight norm << buildings, some sensitivity remains).")
        else:
            print("VERDICT: walls_lidar IS wired through env_net (not completely discarded).")
            print("       Bottleneck may still under-emphasize walls vs other slices.")


if __name__ == "__main__":
    main()
