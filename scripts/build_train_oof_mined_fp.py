from __future__ import annotations

import argparse
from pathlib import Path

from kiloc.mining import (
    build_mined_false_positive_rows,
    discover_relation_csvs,
    summarize_mined_false_positive_rows,
    write_mined_false_positive_csv,
    write_mined_false_positive_summary,
)
from kiloc.oof import read_relation_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate kept-for-mining OOF candidates into one train_oof_mined_fp.csv table."
    )
    parser.add_argument("--oof-run-dir", required=True, type=Path)
    parser.add_argument("--out-csv", default=None, type=Path)
    parser.add_argument("--out-summary", default=None, type=Path)
    parser.add_argument("--fold-indices", nargs="*", type=int, default=None)
    parser.add_argument("--mining-tag", default="oof_kept_for_mining")
    args = parser.parse_args()

    relation_csvs = discover_relation_csvs(
        args.oof_run_dir,
        fold_indices=args.fold_indices,
    )

    relation_rows = []
    for fold_index, relation_csv_path in relation_csvs:
        rows = read_relation_csv(relation_csv_path)
        if not rows:
            raise RuntimeError(f"Relation CSV is empty for fold {fold_index}: {relation_csv_path}")
        relation_rows.extend(rows)

    mined_rows = build_mined_false_positive_rows(
        relation_rows,
        mining_tag=args.mining_tag,
    )
    summary = summarize_mined_false_positive_rows(
        mined_rows,
        relation_csvs=relation_csvs,
    )
    summary["oof_run_dir"] = str(args.oof_run_dir)
    summary["mining_tag"] = args.mining_tag

    out_csv = args.out_csv if args.out_csv is not None else args.oof_run_dir / "train_oof_mined_fp.csv"
    out_summary = (
        args.out_summary
        if args.out_summary is not None
        else args.oof_run_dir / "train_oof_mined_fp_summary.json"
    )

    write_mined_false_positive_csv(mined_rows, out_csv)
    write_mined_false_positive_summary(summary, out_summary)

    print(f"relation_csvs={len(relation_csvs)}")
    print(f"num_mined_rows={summary['num_mined_rows']}")
    print(f"num_fold_image_pairs={summary['num_fold_image_pairs']}")
    print(f"saved_csv={out_csv}")
    print(f"saved_summary={out_summary}")


if __name__ == "__main__":
    main()
