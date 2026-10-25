import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rag_hpo.embedding_model import EmbeddingModel
from rag_hpo.utils import _clean_text


class Embedder:
    _NEG_PATTERN = re.compile(
        r"\b(?:decreas(?:e|ed|ing)?|loss(?:es)?|hypo-?\w+)\b", re.IGNORECASE
    )
    _POS_PATTERN = re.compile(
        r"\b(?:increas(?:e|ed|ing)?|gain(?:s|ed)?|hyper-?\w+)\b", re.IGNORECASE
    )

    _CONSTANT_FIELDS: tuple[str, ...] = (
        "organ_system",
        "lineage",
        "definition",
        "alt_ids",
        "snomedct",
        "umls",
    )

    def __init__(self, model: EmbeddingModel, batch_size: int = 64) -> None:
        self._model = model
        self._batch_size = batch_size

    def get_embedding(
        self, hpo_df: pd.DataFrame, hp_meta_file: Path, emb_file: Path
    ) -> None:
        constants: dict[str, dict[str, Any]] = {}
        entries: list[dict[str, Any]] = []
        infos: list[str] = []

        for row in hpo_df.to_dict(orient="records"):
            info = _clean_text(row["phrase"])
            infos.append(info)
            entries.append(
                {
                    "hp_id": row["hp_id"],
                    "info": info,
                    "direction": self._direction(info),
                }
            )
            if row["hp_id"] not in constants:
                constants[row["hp_id"]] = {
                    f: getattr(row, f) for f in self._CONSTANT_FIELDS
                }

        if not infos:
            raise ValueError("hpo_df is empty; nothing to embed")

        emb_matrix = self._model.encode(infos, batch_size=self._batch_size).astype(
            np.float16
        )

        combined = {
            "model": self._model.model_name,
            "constants": constants,
            "entries": entries,
        }
        with open(hp_meta_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, separators=(",", ":"))

        np.savez_compressed(emb_file, emb=emb_matrix)
        print(f"Saved {len(entries)} embeddings → {hp_meta_file}, {emb_file}")

    def _direction(self, info: str) -> int:
        if self._NEG_PATTERN.search(info):
            return -1
        if self._POS_PATTERN.search(info):
            return 1
        return 0
