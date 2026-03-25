from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from kiloc.utils.config import get_paths


FOLD_HOLDOUT_RE = re.compile(r"^fold_(\d+)_holdout_images\.txt$")


def discover_fold_indices(fold_dir: Path) -> list[int]:
    fold_indices: list[int] = []
    for path in fold_dir.iterdir():
        match = FOLD_HOLDOUT_RE.match(path.name)
        if match is not None:
            fold_indices.append(int(match.group(1)))

    if not fold_indices:
        raise RuntimeError(f"No fold holdout manifests found in {fold_dir}")

    return sorted(set(fold_indices))


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def run_one_fold(
    *,
    fold_index: int,
    gpu_id: int,
    config_path: Path,
    fold_dir: Path,
    group_out_dir: str,
    device_key: str,
    logs_dir: Path,
) -> dict:
    run_name = f"fold_{fold_index}"
    log_path = logs_dir / f"{run_name}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable,
        "scripts/run_oof_fold.py",
        "--config",
        str(config_path),
        "--fold-dir",
        str(fold_dir),
        "--fold-index",
        str(fold_index),
        "--out-dir",
        group_out_dir,
        "--run-name",
        run_name,
        "--device-key",
        device_key,
    ]

    run_dir = None
    with log_path.open("w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            log_f.write(line)
            log_f.flush()
            if line.startswith("RUN_DIR:"):
                run_dir = Path(line.strip().split(":", 1)[1])

        process.wait()

    if process.returncode != 0:
        return {
            "fold_index": fold_index,
            "gpu_id": gpu_id,
            "status": "failed",
            "returncode": process.returncode,
            "log_path": str(log_path),
        }

    if run_dir is None:
        return {
            "fold_index": fold_index,
            "gpu_id": gpu_id,
            "status": "failed",
            "reason": "RUN_DIR not found",
            "log_path": str(log_path),
        }

    holdout_results_path = run_dir / f"fold_{fold_index}_holdout_results.json"
    metadata_path = run_dir / "oof_metadata.json"
    history_path = run_dir / "history.json"

    with holdout_results_path.open("r", encoding="utf-8") as f:
        holdout_results = json.load(f)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    best_history = max(history, key=lambda row: row["f1_macro"])

    return {
        "fold_index": fold_index,
        "gpu_id": gpu_id,
        "status": "ok",
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "best_epoch": metadata["best_epoch"],
        "best_val_f1_macro": metadata["best_val_f1_macro"],
        "holdout_f1_macro": holdout_results["f1_macro"],
        "holdout_precision": holdout_results["precision"],
        "holdout_recall": holdout_results["recall"],
        "holdout_f1": holdout_results["f1"],
        "holdout_f1_pos": holdout_results["f1_pos"],
        "holdout_f1_neg": holdout_results["f1_neg"],
        "num_prediction_rows": holdout_results["num_prediction_rows"],
        "num_train_images": metadata["num_train_images"],
        "num_holdout_images": metadata["num_holdout_images"],
        "history_best_epoch": best_history["epoch"],
    }


def main(
    config_path: Path,
    fold_dir: Path,
    gpus: list[int],
    out_dir: str,
    run_group: str | None,
    device_key: str,
    fold_indices: list[int] | None,
) -> None:
    if not gpus:
        raise ValueError("At least one GPU id must be provided.")

    discovered_folds = discover_fold_indices(fold_dir)
    if fold_indices is None or len(fold_indices) == 0:
        fold_indices = discovered_folds
    else:
        unknown = sorted(set(fold_indices).difference(discovered_folds))
        if unknown:
            raise ValueError(f"Requested fold indices not found in {fold_dir}: {unknown}")
        fold_indices = sorted(set(fold_indices))

    _, checkpoint_dir = get_paths(device=device_key)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_group is None:
        run_group = f"oofrun_{timestamp}"

    group_out_dir = str(Path(out_dir) / run_group)
    group_dir = checkpoint_dir / group_out_dir
    logs_dir = group_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"OOF fold launcher: {len(fold_indices)} folds")
    print(f"Folds: {fold_indices}")
    print(f"GPUs: {gpus}")
    print(f"Output group: {group_dir}")

    results: list[dict] = []
    futures = []

    with ThreadPoolExecutor(max_workers=min(len(gpus), len(fold_indices))) as ex:
        for i, fold_index in enumerate(fold_indices):
            gpu_id = gpus[i % len(gpus)]
            futures.append(
                ex.submit(
                    run_one_fold,
                    fold_index=fold_index,
                    gpu_id=gpu_id,
                    config_path=config_path,
                    fold_dir=fold_dir,
                    group_out_dir=group_out_dir,
                    device_key=device_key,
                    logs_dir=logs_dir,
                )
            )

        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)

            if result["status"] == "ok":
                print(
                    f"[GPU {result['gpu_id']}] "
                    f"fold={result['fold_index']} "
                    f"val_F1_macro={result['best_val_f1_macro']:.4f} "
                    f"holdout_F1_macro={result['holdout_f1_macro']:.4f} "
                    f"best_epoch={result['best_epoch']}"
                )
            else:
                print(f"[GPU {result['gpu_id']}] FAILED {result}")

    ok_results = [result for result in results if result["status"] == "ok"]
    ok_results.sort(key=lambda result: result["fold_index"])

    aggregate = None
    if ok_results:
        aggregate = {
            "num_completed_folds": len(ok_results),
            "mean_holdout_f1_macro": safe_mean([result["holdout_f1_macro"] for result in ok_results]),
            "mean_holdout_f1": safe_mean([result["holdout_f1"] for result in ok_results]),
            "mean_holdout_precision": safe_mean([result["holdout_precision"] for result in ok_results]),
            "mean_holdout_recall": safe_mean([result["holdout_recall"] for result in ok_results]),
            "mean_holdout_f1_pos": safe_mean([result["holdout_f1_pos"] for result in ok_results]),
            "mean_holdout_f1_neg": safe_mean([result["holdout_f1_neg"] for result in ok_results]),
            "total_prediction_rows": sum(result["num_prediction_rows"] for result in ok_results),
        }

    summary = {
        "config_path": str(config_path),
        "fold_dir": str(fold_dir),
        "fold_indices": fold_indices,
        "gpus": gpus,
        "device_key": device_key,
        "group_dir": str(group_dir),
        "results": ok_results,
        "aggregate": aggregate,
        "failed": [result for result in results if result["status"] != "ok"],
    }

    output_path = group_dir / "oof_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== OOF RESULTS ===")
    for result in ok_results:
        print(
            f"fold={result['fold_index']} "
            f"| gpu={result['gpu_id']} "
            f"| val_F1_macro={result['best_val_f1_macro']:.4f} "
            f"| holdout_F1_macro={result['holdout_f1_macro']:.4f} "
            f"| epoch={result['best_epoch']}"
        )

    if aggregate is not None:
        print(
            "\nAggregate: "
            f"mean_holdout_F1_macro={aggregate['mean_holdout_f1_macro']:.4f}, "
            f"mean_holdout_F1={aggregate['mean_holdout_f1']:.4f}, "
            f"mean_holdout_P={aggregate['mean_holdout_precision']:.4f}, "
            f"mean_holdout_R={aggregate['mean_holdout_recall']:.4f}"
        )

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch all OOF fold trainings across one or more GPUs."
    )
    parser.add_argument("--config", default="configs/train_mse.yaml", type=Path)
    parser.add_argument("--fold-dir", required=True, type=Path)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--out-dir", default="oof")
    parser.add_argument("--run-group", default=None)
    parser.add_argument("--device-key", default="h200", choices=["hpvictus", "collab", "h200"])
    parser.add_argument("--fold-indices", nargs="*", type=int, default=None)
    args = parser.parse_args()

    main(
        config_path=args.config,
        fold_dir=args.fold_dir,
        gpus=args.gpus,
        out_dir=args.out_dir,
        run_group=args.run_group,
        device_key=args.device_key,
        fold_indices=args.fold_indices,
    )
