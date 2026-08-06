# SAR (search-and-rescue) — paper protocol

Train a **single-agent** RCO policy on `MASAR1WC`, then deploy the **shared** checkpoint on `MASAR2WC` with a central Büchi coordinator (paper §5.3). Inter-agent collision is **not** modeled in the cost/termination stack.

Assumes this repo is checked out under `RISE-2026/` next to `RISE-Training/` and `SpecRLBench/`.

```bash
# Train (solo, SyncEnv — GenZ default)
PYTHONPATH=src/ python run_sar.py --script train_rco --name GenZ-SAR \
  --env PointLTL0MASAR1WC-v0 --curriculum PointLTL0MASAR1WC-v0 \
  --model_config PointLTL0MASAR1WC-v0 --num_seeds 100 --num_procs 1

# Fast async train (RISE-owned vec)
python ../RISE-Training/train/genz_rco_async.py \
  --name GenZ-SAR-s0 --env PointLTL0MASAR1WC-v0 \
  --curriculum PointLTL0MASAR1WC-v0 --model_config PointLTL0MASAR1WC-v0 \
  --seed 0 --num_procs 24

# Single-agent LTL eval (MASAR1WC checkpoint)
PYTHONPATH=src/ python src/evaluation/simulate.py --env PointLTL0MASAR1WC-v0 \
  --exp GenZ-SAR --seed 0 --formula "(!surface_0 U entrapped_0) & F surface_0"

# Two-agent deploy eval (shared MASAR1WC checkpoint + coordinator on MASAR2WC)
cd ../RISE-Training
python eval_genz_ma_sar.py --exp GenZ-SAR --seed 0 \
  --formula "(!(any_walls | any_surface) U all_entrapped) & (!any_walls U all_surface)"

# Zone pretrained → SAR transfer (48-d zone_compat features; experimental)
python eval_genz_ma_sar.py \
  --train_env PointLtlSafety2-v0 --eval_env PointLTL0MASAR2WC-v0 \
  --exp GenZ-LTL --seed 1 --zone-compat \
  --formula "(!(any_walls | any_surface) U all_entrapped) & (!any_walls U all_surface)"
```

Requires `experiments/rco/PointLtlSafety2-v0/GenZ-LTL/<seed>/status.pth` for zone→SAR transfer. `zone_compat` drops always-on buildings/walls lidars, maps entrapped reach/avoid lidar to buildings, and feeds walls into the avoid channel when the avoid set includes `walls` / `any_walls`.

SafePO uses the same train→deploy split; see `RISE-Training/rise_training/cmdp/ma_protocol.md`.

Multi-agent **MARL** baselines (e.g. SafePO IPPO on `PointLTL0MASAR2-v0`) are optional comparisons in `RISE-Training`, not the paper protocol.

Or from the monorepo root: `./scripts/sar_genz_sa_to_ma.sh`
