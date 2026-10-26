import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pronto

from rag_hpo.embedding_model import EmbeddingModel
from rag_hpo._hpo_df_builder import HPODataframeBuilder
from rag_hpo.ontology import Ontology
from rag_hpo.utils import _clean_text

logger = logging.getLogger(__name__)


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

    def build_vector_db(
        self, ontology: pronto.Ontology, hp_meta_file: Path, emb_file: Path
    ) -> None:
        """
        Embed HPO term phrases and write the vector database to disk.

        Each row of ``hpo_df`` is one phrase (a term label or synonym) for an HPO
        term. Phrases are cleaned, embedded in batches, and saved as two files
        that ``RAGHPO`` loads at query time:

        - ``hp_meta_file`` (JSON): the model name, per-term constants (one entry
          per unique ``hp_id``, holding the fields in ``_CONSTANT_FIELDS``), and
          per-phrase entries (``hp_id``, cleaned ``info`` text, ``direction``).
        - ``emb_file`` (compressed .npz): an ``emb`` array of float16 embeddings,
          one row per entry, in the same order as ``entries``.

        Args:
            ontology: A loaded HPO instance as a pronto.Ontology.
            hp_meta_file: Path to write the metadata JSON to. Overwritten if it exists.
            emb_file: Path to write the embeddings to. Overwritten if it exists.
                Should end in ``.npz``; otherwise numpy appends the suffix.

        Raises:
            ValueError: If ``hpo_df`` is empty.
            KeyError: If ``hpo_df`` is missing a required column.
        """
        constants: dict[str, dict[str, Any]] = {}
        entries: list[dict[str, Any]] = []
        infos: list[str] = []

        ontology = Ontology(ontology)
        hpo_df = HPODataframeBuilder(ontology).build_hpo_dataframe()

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
                constants[row["hp_id"]] = {f: row[f] for f in self._CONSTANT_FIELDS}

        if not infos:
            raise ValueError("hpo_df is empty; nothing to embed")

        logger.debug(
            "Encoding %d phrases (%d unique HPO terms) with %s, batch_size=%d",
            len(infos),
            len(constants),
            self._model.model_name,
            self._batch_size,
        )
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
        logger.info(
            "Saved %d embeddings → %s, %s", len(entries), hp_meta_file, emb_file
        )

    def _direction(self, info: str) -> int:
        if self._NEG_PATTERN.search(info):
            return -1
        if self._POS_PATTERN.search(info):
            return 1
        return 0
