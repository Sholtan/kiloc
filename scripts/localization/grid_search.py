import yaml
import json
import subprocess
import itertools
import os
from pathlib import Path
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_one(i, combo, param_names, base_cfg, gridrun_name, grid_runs_dir, gpu_id):
    run_cfg = base_cfg.copy()
    for name, value in zip(param_names, combo):
        run_cfg[name] = value

    run_name = "_".join(f"{k}_{v}" for k, v in zip(param_names, combo))
    full_run_name = f"{gridrun_name}/{run_name}"

    temp_config_path = grid_runs_dir / f"run_{i:03d}.yaml"
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config_path, "w") as f:
        yaml.safe_dump(run_cfg, f)

    log_dir = Path("checkpoints") / gridrun_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{i:03d}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        "python", "scripts/localization/run_train.py",
        "--config", str(temp_config_path),
        "--out_dir", gridrun_name,
        "--run_name", run_name,
    ]

    run_dir = None
    with open(log_path, "w") as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )

        for line in process.stdout:
            log_f.write(line)
            log_f.flush()
            if line.startswith("RUN_DIR:"):
                run_dir = Path(line.strip().split(":", 1)[1])

        process.wait()

    if process.returncode != 0:
        return {
            "params": dict(zip(param_names, combo)),
            "gpu_id": gpu_id,
            "status": "failed",
            "returncode": process.returncode,
            "log_path": str(log_path),
        }

    if run_dir is None:
        return {
            "params": dict(zip(param_names, combo)),
            "gpu_id": gpu_id,
            "status": "failed",
            "reason": "RUN_DIR not found",
            "log_path": str(log_path),
        }

    with open(run_dir / "history.json") as f:
        history = json.load(f)

    best = max(history, key=lambda x: x["f1_macro"])

    return {
        "params": dict(zip(param_names, combo)),
        "gpu_id": gpu_id,
        "status": "ok",
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "best_epoch": best["epoch"],
        "f1_macro": best["f1_macro"],
        "precision": best["precision"],
        "recall": best["recall"],
    }


def main(grid_config_path, gpus):
    with open(grid_config_path) as f:
        grid_cfg = yaml.safe_load(f)

    with open(grid_cfg["base_config"]) as f:
        base_cfg = yaml.safe_load(f)

    grid = grid_cfg.pop("grid")
    grid_cfg.pop("base_config")
    base_cfg.update(grid_cfg)

    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(itertools.product(*param_values))

    print(f"Grid search: {len(combinations)} runs")
    print(f"Parameters: {param_names}")
    print(f"GPUs: {gpus}")

    grid_runs_dir = Path("configs/grid_runs")
    grid_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gridrun_name = f"gridrun_{grid_timestamp}"

    results = []

    futures = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        for i, combo in enumerate(combinations):
            gpu_id = gpus[i % len(gpus)]   # round-robin assignment
            futures.append(
                ex.submit(
                    run_one,
                    i, combo, param_names, base_cfg,
                    gridrun_name, grid_runs_dir, gpu_id
                )
            )

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)

            if res["status"] == "ok":
                print(
                    f"[GPU {res['gpu_id']}] "
                    f"F1_macro={res['f1_macro']:.4f} "
                    f"epoch={res['best_epoch']} "
                    f"{res['params']}"
                )
            else:
                print(f"[GPU {res['gpu_id']}] FAILED {res}")

    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda x: x["f1_macro"], reverse=True)

    print("\n=== GRID SEARCH RESULTS ===")
    for r in ok_results:
        print(
            f"F1_macro={r['f1_macro']:.4f} "
            f"P={r['precision']:.3f} "
            f"R={r['recall']:.3f} "
            f"| epoch={r['best_epoch']} "
            f"| gpu={r['gpu_id']} "
            f"| {r['params']}"
        )

    output_path = Path("checkpoints") / gridrun_name / "grid_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ok_results, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_config", default="configs/grid_1.yaml")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    args = parser.parse_args()
    main(args.grid_config, args.gpus)