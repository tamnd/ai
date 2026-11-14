# SPDX-License-Identifier: MIT
"""Run an epsilon sweep using the neighboring cliff task and tabular implementation."""
import importlib.util
import json
import argparse
from pathlib import Path
import sys
import numpy as np

cliff = Path(__file__).resolve().parents[1] / "cliff-walking" / "run.py"
spec = importlib.util.spec_from_file_location("cliff_run", cliff)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

parser = argparse.ArgumentParser()
parser.add_argument("--job")
parser.add_argument("--out")
args = parser.parse_args()
if args.job:
    with open(args.job) as f:
        job = json.load(f)
    config = job["config"]
    online_return, reaches_goal, edge_path = module.run_seed(
        int(job["seed"]), "q_learning",
        episodes=config["episodes"],
        max_steps=config["max_steps_per_episode"],
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
    )
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps({
        "final": {
            "online_return": online_return,
            "greedy_reaches_goal_fraction": float(reaches_goal),
            "greedy_edge_path_fraction": float(edge_path),
        },
        "device": "cpu",
    }) + "\n")
    sys.exit(0)

result = {"task": "four-by-twelve cliff walking", "episodes": 150, "return_window": 50, "seeds": list(range(10)), "arms": {}}
for epsilon in (0.0, 0.1, 0.3):
    samples = [module.run_seed(seed, "q_learning", epsilon=epsilon) for seed in range(10)]
    result["arms"][str(epsilon)] = {
        "online_return_mean": float(np.mean([x[0] for x in samples])),
        "online_return_sd": float(np.std([x[0] for x in samples], ddof=1)),
        "greedy_reaches_goal_fraction": float(np.mean([x[1] for x in samples])),
        "greedy_edge_path_fraction": float(np.mean([x[2] for x in samples])),
    }
Path(__file__).with_name("sweep.json").write_text(json.dumps(result, indent=2) + "\n")
