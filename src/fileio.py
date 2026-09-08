"""Crash-safe file writes: write to a temp file in the same folder, then atomically swap it in."""
import json
import os
import tempfile


def atomic_write_text(path: str, text: str, encoding: str = "utf-8"):
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)          # atomic: readers see the old file or the new one, never half
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: str, obj):
    atomic_write_text(path, json.dumps(obj, indent=2))


def atomic_write_jsonl(path: str, records):
    # The whole string is built BEFORE any file is touched, so a bad record can't leave a partial file.
    atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in records))


def atomic_write_faiss(index, path: str):
    import faiss
    tmp = path + ".tmp"
    faiss.write_index(index, tmp)
    os.replace(tmp, path)