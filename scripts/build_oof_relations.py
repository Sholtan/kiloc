from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from kiloc.oof import (
    build_relation_rows,
    load_gt_by_image_from_image_paths,
    read_raw_prediction_csv,
    relation_artifact_paths,
    summarize_relation_rows,
    write_relation_csv,
    write_relation_summary,
)
from kiloc.utils.config import get_paths


TAG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _resolve_fold_index(run_dir: Path, raw_csv_path: Path) -> int:
    with raw_csv_path.open("r", encoding="utf-8") as f:
        header = f.readline()
        first_data = f.readline().strip()

    if first_data:
        return int(first_data.split(",", 1)[0])

    metadata_path = run_dir / "oof_metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        return int(metadata["fold_index"])

    raise RuntimeError("Could not resolve fold index from raw prediction CSV or metadata.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build prediction-to-GT relation table for one OOF fold run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device-key", default="h200", choices=["hpvictus", "collab", "h200"])
    parser.add_argument("--matching-radius", type=float, default=None)
    parser.add_argument("--tau-mine-pos", type=float, default=None)
    parser.add_argument("--tau-mine-neg", type=float, default=None)
    parser.add_argument("--r-exclude-any", type=float, default=None)
    parser.add_argument("--r-cluster-same", type=float, default=None)
    parser.add_argument("--r-interclass-conflict", type=float, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    if args.tag is not None and not TAG_RE.fullmatch(args.tag):
        raise ValueError("tag must match [A-Za-z0-9_.-]+")

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_csv_paths = sorted(run_dir.glob("fold_*_raw_predictions.csv"))
    if len(raw_csv_paths) != 1:
        raise RuntimeError(f"Expected exactly one raw prediction CSV in {run_dir}, found {raw_csv_paths}")
    raw_csv_path = raw_csv_paths[0]
    fold_index = _resolve_fold_index(run_dir, raw_csv_path)

    holdout_manifest = run_dir / f"fold_{fold_index}_holdout_images.txt"
    if not holdout_manifest.exists():
        raise FileNotFoundError(f"Missing holdout manifest: {holdout_manifest}")

    with holdout_manifest.open("r", encoding="utf-8") as f:
        holdout_image_paths = [line.strip() for line in f if line.strip()]

    raw_rows = read_raw_prediction_csv(raw_csv_path)
    data_root, _ = get_paths(device=args.device_key)
    gt_by_image = load_gt_by_image_from_image_paths(data_root, holdout_image_paths)

    base_threshold = float(cfg.get("threshold", 0.5))
    tau_mine = (
        float(args.tau_mine_pos if args.tau_mine_pos is not None else cfg.get("thr_pos", base_threshold)),
        float(args.tau_mine_neg if args.tau_mine_neg is not None else cfg.get("thr_neg", base_threshold)),
    )
    matching_radius = float(args.matching_radius if args.matching_radius is not None else cfg["matching_radius"])
    r_exclude_any = float(args.r_exclude_any if args.r_exclude_any is not None else 1.5 * matching_radius)
    r_cluster_same = float(args.r_cluster_same if args.r_cluster_same is not None else 0.75 * matching_radius)
    r_interclass_conflict = float(
        args.r_interclass_conflict if args.r_interclass_conflict is not None else matching_radius
    )

    relation_rows = build_relation_rows(
        raw_rows=raw_rows,
        gt_by_image=gt_by_image,
        tau_mine=tau_mine,
        matching_radius=matching_radius,
        r_exclude_any=r_exclude_any,
        r_cluster_same=r_cluster_same,
        r_interclass_conflict=r_interclass_conflict,
    )
    summary = summarize_relation_rows(
        relation_rows,
        total_holdout_images=len(holdout_image_paths),
    )
    summary.update(
        {
            "fold_index": fold_index,
            "tag": args.tag,
            "matching_radius": matching_radius,
            "tau_mine_pos": tau_mine[0],
            "tau_mine_neg": tau_mine[1],
            "r_exclude_any": r_exclude_any,
            "r_cluster_same": r_cluster_same,
            "r_interclass_conflict": r_interclass_conflict,
        }
    )

    relations_path, summary_path = relation_artifact_paths(
        run_dir,
        fold_index=fold_index,
        tag=args.tag,
    )
    write_relation_csv(relation_rows, relations_path)
    write_relation_summary(summary, summary_path)

    print(f"relation_rows={len(relation_rows)}")
    print(f"images_with_predictions={summary['num_images_with_predictions']}")
    print(f"saved_csv={relations_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
