import pickle
from pathlib import Path
from typing import Any

import zstandard as zstd


def save_artifact(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = zstd.ZstdCompressor(level=3).compress(pickle.dumps(obj))
    p.write_bytes(blob)


def load_artifact(path: str | Path) -> Any:
    blob = Path(path).read_bytes()
    return pickle.loads(zstd.ZstdDecompressor().decompress(blob))
