import re

import pandas as pd

from rag_hpo.constants import PHENO_ID
from rag_hpo.ontology import Ontology


class HPODataframeBuilder:
    def __init__(self, ontology: Ontology):
        self._ontology = ontology
        self._pattern = re.compile(r"(.+?):(.+)")
        self._clean_abnormality = re.compile(r"(?i)^Abnormality of(?: the)?\s*")

    def build_hpo_dataframe(self, limit: int | None = None) -> pd.DataFrame:
        records = []
        terms = (
            list(self._ontology.terms())[:limit]
            if limit
            else list(self._ontology.terms())
        )

        for term in terms:
            hp_id = term.id
            label = term.name
            definition = term.definition or ""
            synonyms = [syn.description for syn in term.synonyms]

            # ── ALT IDs ──
            alt_ids = self._sort_by_numeric(list(term.alternate_ids))

            # ── XREFS ──
            snomedct, umls = [], []
            for xr in term.xrefs:
                txt = str(xr)
                m = self._pattern.search(txt)
                ent = m.group(1) if m else txt
                pre, _, _ = ent.partition(":")
                if pre.upper() == "UMLS":
                    umls.append(ent)
                elif pre.upper().startswith("SNOMED"):
                    snomedct.append(ent)
            snomedct = self._sort_by_numeric(snomedct)
            umls = self._sort_by_numeric(umls)

            # ── LINEAGE ──
            paths = self._ontology.root_to_node_paths(hp_id) or [
                [self._ontology.root_id(), hp_id]
            ]
            for path_ids in paths:
                lineage_str = " -> ".join(
                    f"{self._ontology.get_label(i)} ({i})" for i in path_ids
                )

                # ── ORGAN SYSTEM ──
                if PHENO_ID in path_ids:
                    idx = path_ids.index(PHENO_ID)
                    organ = (
                        self._ontology.get_label(path_ids[idx + 1]) or "Other"
                        if idx + 1 < len(path_ids)
                        else "Other"
                    )
                else:
                    organ = "Other"
                organ_system = self._clean_abnormality.sub("", organ).title()

                # ── EMIT ROWS ──
                for phrase in [label] + synonyms:
                    if not phrase:
                        continue
                    # **Only title-case** the phrase; do NOT strip anything from it
                    clean_phrase = phrase.title()

                    records.append(
                        {
                            "hp_id": hp_id,
                            "phrase": clean_phrase,
                            "organ_system": organ_system,
                            "lineage": lineage_str,
                            "definition": definition,
                            "alt_ids": ";".join(alt_ids),
                            "snomedct": ";".join(snomedct),
                            "umls": ";".join(umls),
                        }
                    )

        return pd.DataFrame(
            records,
            columns=[
                "hp_id",
                "phrase",
                "organ_system",
                "lineage",
                "definition",
                "alt_ids",
                "snomedct",
                "umls",
            ],
        )

    @staticmethod
    def _sort_by_numeric(entries: list[str]) -> list[str]:
        def key_fn(e: str):
            digs = "".join(filter(str.isdigit, e))
            return int(digs) if digs else float("inf")

        return sorted(entries, key=key_fn)
