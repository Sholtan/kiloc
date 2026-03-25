__all__ = ["BCDataDataset"]


def __getattr__(name: str):
    if name == "BCDataDataset":
        from .datasets import BCDataDataset

        return BCDataDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
