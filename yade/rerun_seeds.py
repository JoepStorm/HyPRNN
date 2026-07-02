"""Re-run specific seeds for a config and patch results.json in-place."""

import json
import argparse
from pathlib import Path
from batch_rve_generation_3D_wet import (
    CONFIGS, DAMPING, FRICTION_ANGLE_LARGE, FRICTION_ANGLE_SMALL,
    run_wet_simulation, OUTPUT_DIR,
)
from woodchip_volume_fraction_3D import compute_volume_fraction

p = argparse.ArgumentParser()
p.add_argument("--config", type=str, required=True, help="Config name, e.g. small075_large025")
p.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to re-run, e.g. 0 2 4")
p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR / "mixing_v2"))
args = p.parse_args()

output_dir = Path(args.output_dir)
config = CONFIGS[args.config]
config_dir = output_dir / args.config
results_file = output_dir / "results.json"

with open(results_file) as f:
    all_vfracs = json.load(f)

print(f"Re-running config={args.config}, seeds={args.seeds}")
print(f"Config: {config}")

for seed in args.seeds:
    out = config_dir / f"seed{seed}.npy"
    npy = run_wet_simulation(seed, out, config)
    if npy is None:
        print(f"  seed {seed}: FAILED")
        continue
    vf, _ = compute_volume_fraction(str(npy), method="monte_carlo",
                                    n_samples=500000, periodic=True)
    if vf is not None:
        old = all_vfracs[args.config][seed]
        all_vfracs[args.config][seed] = vf
        print(f"  seed {seed}: VF = {vf:.4f} (was {old:.6f})")
    else:
        print(f"  seed {seed}: VF computation failed")

with open(results_file, "w") as f:
    json.dump(all_vfracs, f, indent=2)
print(f"\nUpdated {results_file}")
