import logging
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


class CheckpointStore:
    """
    Pickle-based checkpoints for named pipeline stages, one file per stage.

    State is a dict mapping stage key -> DataFrame or None, where None means
    "stage not reached yet" and an empty DataFrame means "stage done, no rows".

    The first key marks a run: if its file exists, the directory holds a run
    that can be resumed.

    With directory=None the store is disabled and every operation is a no-op,
    so callers don't need to branch on whether checkpointing is on.
    """

    def __init__(self, directory: Path | str | None, keys: Iterable[str]):
        self.directory = Path(directory) if directory is not None else None
        self.keys = tuple(keys)
        if not self.keys:
            raise ValueError("keys must not be empty.")

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def path(self, key: str) -> Path | None:
        if not self.enabled:
            return None
        return self.directory / f"{key}.pkl"

    def exists(self) -> bool:
        """True if the directory holds a resumable run."""
        p = self.path(self.keys[0])
        return p is not None and p.exists()

    def existing(self) -> list[str]:
        """File names of all checkpoints currently on disk."""
        paths = (self.path(k) for k in self.keys)
        return [p.name for p in paths if p is not None and p.exists()]

    def empty_state(self) -> dict[str, pd.DataFrame | None]:
        return {key: None for key in self.keys}

    def load(self) -> dict[str, pd.DataFrame | None]:
        state = self.empty_state()
        for key in self.keys:
            p = self.path(key)
            if p is not None and p.exists():
                state[key] = pd.read_pickle(p)
                logger.info("Loaded checkpoint '%s' (%d rows)", key, len(state[key]))
        return state

    def save(
        self, state: dict[str, pd.DataFrame | None], keys: Iterable[str] | None = None
    ):
        """Atomically write the given stages (all by default). None entries are skipped."""
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        for key in keys or self.keys:
            df = state.get(key)
            if df is None:
                continue
            p = self.path(key)
            tmp = p.with_suffix(p.suffix + ".tmp")
            try:
                df.to_pickle(tmp)
                os.replace(tmp, p)  # atomic
                logger.debug("Checkpointed '%s' (%d rows) -> %s", key, len(df), p)
            except Exception:
                logger.exception("Error saving checkpoint '%s'", key)
                tmp.unlink(missing_ok=True)

    def clear(self):
        """Delete all checkpoint files (the directory itself is left in place)."""
        for key in self.keys:
            p = self.path(key)
            if p is None:
                continue
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Error removing checkpoint %s: %s", p, e)
