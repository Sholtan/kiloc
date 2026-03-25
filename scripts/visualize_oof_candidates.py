from __future__ import annotations

import argparse
from pathlib import Path

from kiloc.oof import load_gt_by_image_from_image_paths, read_relation_csv
from kiloc.utils.config import get_paths


DEFAULT_VISUAL_FILTER_REASONS = (
    "kept_for_mining",
    "below_score_threshold",
    "near_gt_any",
    "interclass_conflict",
    "sameclass_duplicate",
)


def _resolve_fold_run_dir(run_dir: Path, fold_index: int) -> Path:
    direct_relations_path = run_dir / f"fold_{fold_index}_prediction_relations.csv"
    if direct_relations_path.exists():
        return run_dir

    nested_run_dir = run_dir / f"fold_{fold_index}"
    nested_relations_path = nested_run_dir / f"fold_{fold_index}_prediction_relations.csv"
    if nested_relations_path.exists():
        return nested_run_dir

    raise FileNotFoundError(
        f"Could not find fold_{fold_index}_prediction_relations.csv under {run_dir}"
    )


def _resolve_requested_reasons(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    if args.kept_for_mining:
        requested.append("kept_for_mining")
    if args.below_score_threshold:
        requested.append("below_score_threshold")
    if args.near_gt_any:
        requested.append("near_gt_any")
    if args.interclass_conflict:
        requested.append("interclass_conflict")
    if args.sameclass_duplicate:
        requested.append("sameclass_duplicate")

    if requested:
        return requested
    return list(DEFAULT_VISUAL_FILTER_REASONS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize OOF candidate categories for one fold."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--fold-index", required=True, type=int)
    parser.add_argument("--device-key", default="h200", choices=["hpvictus", "collab", "h200"])
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--max-images-per-reason", "--max-per-reason", dest="max_images_per_reason", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--kept-for-mining", action="store_true")
    parser.add_argument("--below-score-threshold", action="store_true")
    parser.add_argument("--near-gt-any", action="store_true")
    parser.add_argument("--interclass-conflict", action="store_true")
    parser.add_argument("--sameclass-duplicate", action="store_true")
    args = parser.parse_args()

    fold_run_dir = _resolve_fold_run_dir(args.run_dir, args.fold_index)
    relation_csv_path = fold_run_dir / f"fold_{args.fold_index}_prediction_relations.csv"
    if not relation_csv_path.exists():
        raise FileNotFoundError(
            f"Missing relation CSV: {relation_csv_path}. Run build_oof_relations.py first."
        )

    relation_rows = read_relation_csv(relation_csv_path)
    if not relation_rows:
        raise RuntimeError(f"No relation rows found in {relation_csv_path}")

    requested_reasons = _resolve_requested_reasons(args)
    from kiloc.oof.visualization import visualize_relation_candidates

    relation_image_paths = sorted({row["image_path"] for row in relation_rows})
    data_root, _ = get_paths(device=args.device_key)
    gt_by_image = load_gt_by_image_from_image_paths(data_root, relation_image_paths)

    if args.out_dir is None:
        output_dir = fold_run_dir / "candidate_visualizations" / f"fold_{args.fold_index}"
    else:
        output_dir = args.out_dir

    saved_counts = visualize_relation_candidates(
        relation_rows=relation_rows,
        gt_by_image=gt_by_image,
        data_root=data_root,
        output_dir=output_dir,
        filter_reasons=requested_reasons,
        max_images_per_reason=args.max_images_per_reason,
        dpi=args.dpi,
    )

    print(f"output_dir={output_dir}")
    for reason in requested_reasons:
        print(f"{reason}={saved_counts.get(reason, 0)}")


if __name__ == "__main__":
    main()
