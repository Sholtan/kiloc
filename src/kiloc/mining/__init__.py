__all__ = [
    "BCDataHardNegativeDataset",
    "HardNegativeWeightMapGenerator",
    "LocalHardNegativeBCELoss",
    "LocalHardNegativeGeneralizedCrossEntropyLoss",
    "LocalHardNegativeSigmoidMSELoss",
    "build_hardneg_loss",
    "build_mined_false_positive_rows",
    "collate_fn_hardneg",
    "discover_relation_csvs",
    "group_mined_false_positives_by_image",
    "read_mined_false_positive_csv",
    "summarize_grouped_mined_false_positives",
    "summarize_mined_false_positive_rows",
    "train_one_epoch_hardneg",
    "write_mined_false_positive_csv",
    "write_mined_false_positive_summary",
]


def __getattr__(name: str):
    if name in {"BCDataHardNegativeDataset", "collate_fn_hardneg"}:
        from .datasets import BCDataHardNegativeDataset, collate_fn_hardneg

        return {
            "BCDataHardNegativeDataset": BCDataHardNegativeDataset,
            "collate_fn_hardneg": collate_fn_hardneg,
        }[name]

    if name in {
        "group_mined_false_positives_by_image",
        "read_mined_false_positive_csv",
        "summarize_grouped_mined_false_positives",
    }:
        from .io import (
            group_mined_false_positives_by_image,
            read_mined_false_positive_csv,
            summarize_grouped_mined_false_positives,
        )

        return {
            "group_mined_false_positives_by_image": group_mined_false_positives_by_image,
            "read_mined_false_positive_csv": read_mined_false_positive_csv,
            "summarize_grouped_mined_false_positives": summarize_grouped_mined_false_positives,
        }[name]

    if name in {
        "LocalHardNegativeBCELoss",
        "LocalHardNegativeGeneralizedCrossEntropyLoss",
        "LocalHardNegativeSigmoidMSELoss",
        "build_hardneg_loss",
    }:
        from .losses import (
            LocalHardNegativeBCELoss,
            LocalHardNegativeGeneralizedCrossEntropyLoss,
            LocalHardNegativeSigmoidMSELoss,
            build_hardneg_loss,
        )

        return {
            "LocalHardNegativeBCELoss": LocalHardNegativeBCELoss,
            "LocalHardNegativeGeneralizedCrossEntropyLoss": (
                LocalHardNegativeGeneralizedCrossEntropyLoss
            ),
            "LocalHardNegativeSigmoidMSELoss": LocalHardNegativeSigmoidMSELoss,
            "build_hardneg_loss": build_hardneg_loss,
        }[name]

    if name == "HardNegativeWeightMapGenerator":
        from .targets import HardNegativeWeightMapGenerator

        return HardNegativeWeightMapGenerator

    if name == "train_one_epoch_hardneg":
        from .training import train_one_epoch_hardneg

        return train_one_epoch_hardneg

    if name in {
        "build_mined_false_positive_rows",
        "discover_relation_csvs",
        "summarize_mined_false_positive_rows",
        "write_mined_false_positive_csv",
        "write_mined_false_positive_summary",
    }:
        from .tables import (
            build_mined_false_positive_rows,
            discover_relation_csvs,
            summarize_mined_false_positive_rows,
            write_mined_false_positive_csv,
            write_mined_false_positive_summary,
        )

        return {
            "build_mined_false_positive_rows": build_mined_false_positive_rows,
            "discover_relation_csvs": discover_relation_csvs,
            "summarize_mined_false_positive_rows": summarize_mined_false_positive_rows,
            "write_mined_false_positive_csv": write_mined_false_positive_csv,
            "write_mined_false_positive_summary": write_mined_false_positive_summary,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
