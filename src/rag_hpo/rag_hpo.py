import json
import logging
import os
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rapidfuzz import fuzz
from tabulate import tabulate
from tqdm import tqdm

from rag_hpo.embedder import Embedder
from rag_hpo.embedding_model import EmbeddingModel
from rag_hpo.hpo_df_builder import HPODataframeBuilder
from rag_hpo.hpo_doc import HPODoc
from rag_hpo.llm_client import LLMClient
from rag_hpo.ontology import Ontology
from utils import (
    _JSON_OBJECT_PATTERN,
    _NON_ASCII_PATTERN,
    _WHITESPACE_PATTERN,
    _WORD_PATTERN,
    _clean_text,
)

logger = logging.getLogger(__name__)

# Pipeline stages that get checkpointed (in order).
_CHECKPOINT_KEYS = ("input", "processed", "combined", "exact", "non_exact", "final")
# Columns every per-note findings frame must have.
_FINDING_COLS = ["phrase", "category", "unique_metadata", "original_sentence", "patient_id"]


class RAGHPO:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        meta_path: Path | str = "hpo_meta.json",
        vec_path: Path | str = "hpo_embedded.npz",
    ):
        self._embedding_model = embedding_model
        self._llm_client = llm_client
        self._meta_path = Path(meta_path)
        self._vec_path = Path(vec_path)

        # Loaded lazily on first extract() call.
        self._docs: list[dict[str, Any]] | None = None
        self._index: faiss.Index | None = None
        self._cluster_index: defaultdict | None = None

    @classmethod
    def inti_and_create_embeddings(
        cls,
        ontology: Ontology,
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        hp_meta_file: Path,
        emb_file: Path,
    ) -> RAGHPO:
        df_builder = HPODataframeBuilder(ontology)
        hpo_df = df_builder.build_hpo_dataframe()
        embedder = Embedder(embedding_model)
        embedder.get_embedding(hpo_df, hp_meta_file, emb_file)
        RAGHPO(embedding_model, llm_client, hp_meta_file, emb_file)

    # ======================= Entry points =======================

    def extract(
        self,
        notes: pd.DataFrame | Iterable[str],
        system_message_extract: str,
        system_message_map: str,
        checkpoint_dir: Path | str | None = None,
        notes_checkpoint_every: int = 20,
        mapping_checkpoint_every: int = 50,
    ) -> pd.DataFrame:
        """
        Run the full HPO extraction pipeline on new notes.

        notes:
            DataFrame with a 'clinical_note' column (optionally 'patient_id'),
            or a list of note strings.
        system_message_extract:
            System prompt for phenotype extraction from notes (notebook: system_message_I).
        system_message_map:
            System prompt for mapping phrases to HPO candidates (notebook: system_message_II).
        checkpoint_dir:
            If given, progress is checkpointed there so a failed run can be
            continued with extract_from_checkpoint(). Must not already hold
            checkpoints from another run. Checkpoints are deleted on success.

        Returns a DataFrame with columns ['patient_id', 'HPO_Terms'], where
        HPO_Terms is a list of {'phrase', 'category', 'HPO_Term'} dicts.
        """
        if notes is None:
            raise ValueError("notes is required; use extract_from_checkpoint() to resume.")

        paths = self._checkpoint_paths(checkpoint_dir)
        existing = [p.name for p in paths.values() if p.exists()]
        if existing:
            raise FileExistsError(
                f"{checkpoint_dir} already holds checkpoints ({', '.join(existing)}). "
                "Use extract_from_checkpoint() to resume that run, or delete them first."
            )

        if not isinstance(notes, pd.DataFrame):
            notes = pd.DataFrame({"clinical_note": list(notes)})
        state = {key: None for key in _CHECKPOINT_KEYS}
        state["input"] = self._validate_input(notes)
        self._save_state(state, paths, ["input"])

        return self._run_pipeline(
            state,
            paths,
            system_message_extract,
            system_message_map,
            notes_checkpoint_every,
            mapping_checkpoint_every,
        )

    def extract_from_checkpoint(
        self,
        checkpoint_dir: Path | str,
        system_message_extract: str,
        system_message_map: str,
        notes_checkpoint_every: int = 20,
        mapping_checkpoint_every: int = 50,
    ) -> pd.DataFrame:
        """
        Resume a run that extract() started with the same checkpoint_dir.

        Finished stages are skipped and the notes stage continues from the last
        saved note. Use the same system messages as the original run, otherwise
        results mix outputs from different prompts.

        Returns the same frame as extract().
        """
        if not self.has_checkpoint(checkpoint_dir):
            raise FileNotFoundError(f"No checkpoint to resume from in {checkpoint_dir}.")

        paths = self._checkpoint_paths(checkpoint_dir)
        state = self._load_state(paths)
        logger.info("Resuming from checkpoint in %s", checkpoint_dir)

        return self._run_pipeline(
            state,
            paths,
            system_message_extract,
            system_message_map,
            notes_checkpoint_every,
            mapping_checkpoint_every,
        )

    @staticmethod
    def has_checkpoint(checkpoint_dir: Path | str) -> bool:
        """True if checkpoint_dir holds a run that extract_from_checkpoint() can resume."""
        return (Path(checkpoint_dir) / "input.pkl").exists()

    def run_interactive(
        self,
        system_message_extract: str,
        system_message_map: str,
        checkpoint_dir: Path | str = "checkpoints",
    ) -> pd.DataFrame:
        """Console version of the notebook's main(): prompts for input and output."""
        if self.has_checkpoint(checkpoint_dir):
            final = self.extract_from_checkpoint(
                checkpoint_dir, system_message_extract, system_message_map
            )
        else:
            final = self.extract(
                self._prompt_for_notes(),
                system_message_extract,
                system_message_map,
                checkpoint_dir=checkpoint_dir,
            )
        self._process_results(final)
        return final

    # ======================= Pipeline stages =======================

    def _run_pipeline(
        self,
        state,
        paths,
        system_message_extract,
        system_message_map,
        notes_checkpoint_every,
        mapping_checkpoint_every,
    ) -> pd.DataFrame:
        start = time.time()
        self._ensure_index()

        try:
            self._stage_process_notes(state, paths, system_message_extract, notes_checkpoint_every)
            self._stage_split(state, paths)
            self._stage_map_non_exact(state, paths, system_message_map, mapping_checkpoint_every)
            self._stage_compile(state, paths)
        except Exception:
            logger.exception("Pipeline error. Saving progress for resume.")
            self._save_state(state, paths)
            raise

        self._cleanup(paths)
        logger.info("Pipeline completed in %.2fs", time.time() - start)
        return state["final"]

    def _ensure_index(self):
        if self._index is not None:
            return
        docs, emb_matrix = self._load_vector_db(self._meta_path, self._vec_path)
        self._index = self._create_faiss_index(emb_matrix, metric="cosine")
        self._docs = docs
        self._cluster_index = self._build_cluster_index(docs)

    def _stage_process_notes(self, state, paths, system_message, every):
        inp = state["input"]
        done = set(state["processed"]["row"]) if state["processed"] is not None else set()
        todo = [i for i in range(len(inp)) if i not in done]
        if not todo:
            logger.info("All notes already processed.")
            if state["combined"] is None:
                state["combined"] = pd.DataFrame(columns=_FINDING_COLS)
            return

        parts = [state["combined"]] if state["combined"] is not None else []
        for n, i in enumerate(tqdm(todo, desc="Processing notes", unit="note"), start=1):
            row = inp.iloc[i]
            res = self._process_row(row["clinical_note"], system_message)
            if not res.empty:
                res["patient_id"] = row["patient_id"]
                parts.append(res)
            done.add(i)

            if n % every == 0 or n == len(todo):
                state["combined"] = (
                    pd.concat(parts, ignore_index=True)
                    if parts
                    else pd.DataFrame(columns=_FINDING_COLS)
                )
                state["processed"] = pd.DataFrame({"row": sorted(done)})
                self._save_state(state, paths, ["combined", "processed"])

    def _stage_split(self, state, paths):
        if state["exact"] is not None and state["non_exact"] is not None:
            logger.info("Skipped splitting; checkpoint exists.")
            return

        df = state["combined"].copy()
        if "HPO_Term" not in df.columns:
            df["HPO_Term"] = None
        df["HPO_Term"] = df["HPO_Term"].astype(object)

        if not df.empty:
            df["HPO_Term"] = [
                hp
                if isinstance(hp, str)
                else self._extract_hpo_term(phrase, metadata, self._cluster_index)
                for phrase, metadata, hp in zip(
                    df["phrase"], df["unique_metadata"], df["HPO_Term"]
                )
            ]

        state["exact"], state["non_exact"] = self._split_exact_nonexact(df, "HPO_Term")
        self._save_state(state, paths, ["exact", "non_exact"])

    def _stage_map_non_exact(self, state, paths, system_message, every):
        non_ex = state["non_exact"].copy()
        for col in ("HPO_Term", "llm_parse_reason", "raw_llm_resp"):
            if col not in non_ex.columns:
                non_ex[col] = None
            non_ex[col] = non_ex[col].astype(object)

        todo = non_ex.index[(non_ex["category"] == "Abnormal") & non_ex["HPO_Term"].isna()]
        if todo.empty:
            logger.info("No non-exact entries to process.")
            return

        logger.info("Generating HPO for %d entries...", len(todo))
        for n, idx in enumerate(tqdm(todo, desc="Generating HPO", unit="entry"), start=1):
            rec = self._generate_hpo_terms(non_ex.loc[[idx]], system_message).iloc[0]
            hp = rec["HPO_Terms"][0]["HPO_Term"]
            non_ex.at[idx, "llm_parse_reason"] = rec["llm_parse_reason"]
            non_ex.at[idx, "raw_llm_resp"] = rec["raw_llm_resp"]
            non_ex.at[idx, "HPO_Term"] = hp or "No Candidate Fit"

            if n % every == 0 or n == len(todo):
                state["non_exact"] = non_ex.copy()
                self._save_state(state, paths, ["non_exact"])

    def _stage_compile(self, state, paths):
        if state["final"] is not None:
            logger.info("Skipped final compilation; checkpoint exists.")
            return

        merged = pd.concat([state["exact"], state["non_exact"]], ignore_index=True)
        merged = merged.dropna(subset=["HPO_Term"])
        if merged.empty:
            final = pd.DataFrame(columns=["patient_id", "HPO_Terms"])
        else:
            final = (
                merged.groupby("patient_id")[["phrase", "category", "HPO_Term"]]
                .apply(lambda g: g.to_dict("records"))
                .reset_index(name="HPO_Terms")
            )
        state["final"] = final
        self._save_state(state, paths, ["final"])

    # ======================= Checkpointing =======================

    @staticmethod
    def _checkpoint_paths(checkpoint_dir) -> dict[str, Path]:
        if checkpoint_dir is None:
            return {}
        d = Path(checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        return {key: d / f"{key}.pkl" for key in _CHECKPOINT_KEYS}

    @staticmethod
    def _load_state(paths: dict[str, Path]) -> dict[str, pd.DataFrame | None]:
        # None = stage not reached yet; an empty DataFrame = stage done, no rows.
        state = {key: None for key in _CHECKPOINT_KEYS}
        for key, path in paths.items():
            if path.exists():
                state[key] = pd.read_pickle(path)
                logger.info("Loaded checkpoint '%s' (%d rows)", key, len(state[key]))
        return state

    @staticmethod
    def _save_state(state, paths: dict[str, Path], keys=None):
        for key in keys or _CHECKPOINT_KEYS:
            df, path = state.get(key), paths.get(key)
            if df is None or path is None:
                continue
            tmp = path.with_suffix(path.suffix + ".tmp")
            try:
                df.to_pickle(tmp)
                os.replace(tmp, path)  # atomic
                logger.debug("Checkpointed '%s' (%d rows) -> %s", key, len(df), path)
            except Exception:
                logger.exception("Error saving checkpoint '%s'", key)
                tmp.unlink(missing_ok=True)

    @staticmethod
    def _cleanup(paths: dict[str, Path]):
        for path in paths.values():
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Error removing checkpoint %s: %s", path, e)

    @staticmethod
    def _prompt_for_notes() -> pd.DataFrame:
        if input("Manual notes? (yes/no): ").strip().lower() == "yes":
            notes = []
            while True:
                note = input("Note (or 'done'): ")
                if note.strip().lower() == "done":
                    break
                notes.append(note)
            return pd.DataFrame({"clinical_note": notes})
        while True:
            fname = input("CSV filename: ").strip()
            try:
                return pd.read_csv(fname)
            except Exception as e:
                print(f"Error loading CSV: {e}")

    @staticmethod
    def _process_results(final_df: pd.DataFrame):
        # Handles output of final results (CSV or display)
        if final_df.empty:
            logger.info("No final results to process.")
            return
        choice = input("Save results as CSV or display? (save/display): ").strip().lower()
        if choice == "save":
            fname = input("Output CSV filename (e.g., results.csv): ").strip()
            if not fname:
                print("Filename cannot be empty. Skipping save.")
                return
            rows = []
            for idx, r in final_df.iterrows():
                pid = r.get("patient_id", idx)
                for term in r.get("HPO_Terms", []):
                    ph = term.get("phrase", "").strip()
                    cat = term.get("category", "")
                    hp = term.get("HPO_Term") or ""
                    if isinstance(hp, str):
                        hp = hp.replace("HP:HP:", "HP:")
                    if hp:
                        rows.append(
                            {
                                "Patient ID": pid,
                                "Category": cat,
                                "Phenotype name": ph,
                                "HPO ID": hp,
                            }
                        )
                    else:
                        print(
                            f"Warning: Blank HPO_Term for patient {pid}, phrase '{ph}', "
                            f"category '{cat}' - not included in CSV."
                        )
            if rows:
                pd.DataFrame(rows).to_csv(fname, index=False)
                logger.info("Saved tabular results to %s", fname)
            else:
                logger.info("No valid HPO terms to save in tabular format.")
            raw_name = f"{os.path.splitext(fname)[0]}_json_raw.csv"
            final_df.to_csv(raw_name, index=False)
            logger.info("Saved raw JSON results to %s", raw_name)
        elif choice == "display":
            tbl = []
            for idx, r in final_df.iterrows():
                pid = r.get("patient_id", idx)
                for term in r.get("HPO_Terms", []):
                    hp = term.get("HPO_Term") or ""
                    if isinstance(hp, str):
                        hp = hp.replace("HP:HP:", "HP:")
                    tbl.append(
                        {
                            "Case": f"Case {pid}",
                            "Category": term.get("category", ""),
                            "Phenotype name": term.get("phrase", "").strip(),
                            "HPO ID": hp,
                        }
                    )
            if tbl:
                print(tabulate(pd.DataFrame(tbl), headers="keys", tablefmt="psql"))
            else:
                logger.info("No terms to display.")
        else:
            print("Invalid choice; please enter 'save' or 'display'.")

    # ======================= Helpers =======================

    def _embed_query(self, text: str, metric: str = "cosine"):
        vec = np.ascontiguousarray(
            np.asarray(self._embedding_model.encode(text), dtype=np.float32)
        )
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        if metric == "cosine":
            faiss.normalize_L2(vec)
        return vec

    @staticmethod
    def _clean_note(text: str) -> str:
        text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        text = unicodedata.normalize("NFKD", text)
        text = _WHITESPACE_PATTERN.sub(" ", text)
        text = _NON_ASCII_PATTERN.sub("", text).strip()
        return text

    @staticmethod
    def _split_exact_nonexact(df: pd.DataFrame, hpo_term_col="HPO_Term"):
        if "category" not in df.columns:
            raise KeyError(f"[FATAL] 'category' missing; columns: {df.columns.tolist()}")
        if hpo_term_col not in df.columns:
            raise KeyError(f"[FATAL] '{hpo_term_col}' missing; columns: {df.columns.tolist()}")

        # only keep Abnormal rows
        df_ab = df[df["category"] == "Abnormal"]
        exact_df = df_ab.dropna(subset=[hpo_term_col]).copy()
        non_exact_df = df_ab[df_ab[hpo_term_col].isna()].copy()
        return exact_df, non_exact_df

    @staticmethod
    def _clean_and_parse(s: str):
        # Extracts and parses JSON from string
        try:
            m = _JSON_OBJECT_PATTERN.search(s)
            js_str = m.group(0) if m else s.strip()
            return json.loads(js_str)
        except Exception:
            return None

    @staticmethod
    def _extract_findings(response: str) -> list:
        # Extracts findings from LLM response
        if not response:
            return []
        parsed = RAGHPO._clean_and_parse(response)
        if not isinstance(parsed, dict):
            return []
        return parsed.get("phenotypes", [])

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
        if "clinical_note" not in df.columns:
            raise KeyError("Missing required column: 'clinical_note'.")
        df = df.dropna(subset=["clinical_note"]).copy()
        df["clinical_note"] = df["clinical_note"].astype(str)

        # Clean all clinical notes to prevent encoding-related bugs
        df["clinical_note"] = df["clinical_note"].apply(RAGHPO._clean_note)
        if "patient_id" not in df.columns:
            df = df.reset_index(drop=True)
            df["patient_id"] = df.index + 1
        else:
            df["patient_id"] = df["patient_id"].astype(int)
        return df

    def _generate_hpo_terms(self, df_row: pd.DataFrame, system_message: str) -> pd.DataFrame:
        """
        Streamlined LLM + fallback logic with phrase normalization.
        """
        # 1. Extract & normalize inputs
        phrase = df_row["phrase"].iloc[0].strip()
        normalized = phrase.lower().replace("-", " ").strip()
        category = df_row["category"].iloc[0]
        original = df_row["original_sentence"].iloc[0]
        metadata_list = df_row["unique_metadata"].iloc[0] or []

        # 2. Build candidate list
        candidates = []
        seen = set()
        for m in metadata_list:
            term = m.get("phrase") or m.get("info")
            hp = m.get("hp_id")
            if term and hp and hp not in seen:
                candidates.append({"term": term, "id": hp})
                seen.add(hp)
        candidate_ids = {c["id"] for c in candidates}

        # 3. Call LLM and parse
        payload = json.dumps(
            {
                "phrase": normalized,
                "category": category,
                "original_sentence": original,
                "candidates": candidates,
            }
        )
        resp = self._llm_client.query(payload, system_message)
        hpo_id, reason, _ = self._parse_llm_mapping(resp, candidate_ids)

        # 4. Local fallback if needed
        if not hpo_id:
            cluster_idx = self._build_cluster_index(metadata_list)
            local_id = self._extract_hpo_term(normalized, metadata_list, cluster_idx)
            if local_id:
                hpo_id, reason = local_id, "fallback_local"

        # 5. Return unified record
        return pd.DataFrame(
            [
                {
                    "HPO_Terms": [{"phrase": phrase, "HPO_Term": hpo_id}],
                    "raw_llm_resp": resp,
                    "llm_parse_reason": reason,
                }
            ]
        )

    @staticmethod
    def _parse_llm_mapping(
        resp_text: str, candidate_ids: set
    ) -> tuple[str | None, str, dict | None]:
        """
        Simplified parsing: JSON → key lookup → regex fallback.
        """
        resp_text = resp_text or ""

        # 1. JSON parse (tolerates code fences / surrounding text)
        js = RAGHPO._clean_and_parse(resp_text)

        # 2. Look for an HPO ID in known keys
        if isinstance(js, dict):
            candidate = next(
                (
                    js[k].strip().strip('"')
                    for k in ("hpo_id", "HPO_ID", "hp_id", "id")
                    if isinstance(js.get(k), str)
                ),
                None,
            )
            if candidate:
                low = candidate.lower()
                if low in ("null", "no candidate fit"):
                    return None, "null_label", js
                if candidate in candidate_ids:
                    return candidate, "ok", js
                return candidate, "hp_not_in_candidates", js

        # 3. Regex fallback for HP:NNNNNNN
        for m in re.findall(r"(HP:\d{6,7})", resp_text):
            if m in candidate_ids:
                return m, "regex_fallback", None

        # 4. Nothing found
        return None, "no_hpo_found", None

    @staticmethod
    def _extract_hpo_term(
        phrase: str, metadata_list: list, cluster_index: defaultdict
    ) -> str | None:
        # Maps phenotype phrase to best HPO term using multiple strategies
        if not isinstance(metadata_list, list) or not metadata_list:
            return None
        cp = _clean_text(phrase)
        if not cp:
            return None
        toks = cp.split()
        sig = " ".join(sorted(toks))
        if sig in cluster_index and len(toks) in cluster_index[sig]:
            return cluster_index[sig][len(toks)][0]
        pairs = []
        for entry in metadata_list:
            for term, hp in RAGHPO._iter_term_hp(entry):
                ct = _clean_text(term)
                if ct:
                    pairs.append((ct, hp))
        pset = set(toks)
        for ct, hp in pairs:
            if set(ct.split()) == pset:
                return hp
        for ct, hp in pairs:
            if ct == cp:
                return hp
        if len(pset) > 1:
            for ct, hp in pairs:
                if re.search(rf"\b{re.escape(ct)}\b", cp):
                    return hp
        best_hp, best_score = None, 0
        for ct, hp in pairs:
            score = fuzz.token_sort_ratio(cp, ct)
            if score > best_score:
                best_hp, best_score = hp, score
        return best_hp if best_score >= 80 else None

    @staticmethod
    def _build_cluster_index(metadata_list) -> defaultdict:
        # Builds cluster index for bag-of-words matching
        idx = defaultdict(lambda: defaultdict(list))
        for entry in metadata_list:
            for term, hp in RAGHPO._iter_term_hp(entry):
                ct = _clean_text(term)
                if not ct:
                    continue
                toks = ct.split()
                sig = " ".join(sorted(toks))
                idx[sig][len(toks)].append(hp)
        return idx

    @staticmethod
    def _iter_term_hp(item):
        # Yields (term_text, hp_id) pairs from metadata entries
        if isinstance(item, str):
            try:
                d = json.loads(item)
            except Exception:
                return
        elif isinstance(item, dict):
            d = item
        else:
            return
        if "hp_id" in d:
            # vector-db docs use "info"; retrieved candidates use "phrase"
            term_text = d.get("info") or d.get("label") or d.get("phrase")
            hp = d["hp_id"]
            if hp and term_text:
                yield term_text, hp
            return
        for k, v in d.items():
            if isinstance(v, str) and v.startswith("HP:"):
                yield k, v

    def _process_row(self, clinical_note: str, system_message: str) -> pd.DataFrame:
        # Clean the clinical note before sending to the LLM
        clinical_note = RAGHPO._clean_note(clinical_note)
        raw = self._llm_client.query(clinical_note, system_message)

        # Try to parse findings robustly
        findings = self._extract_findings(raw)
        # Sometimes LLMs return just a list
        if not findings:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    findings = parsed
            except Exception:
                pass

        # Last resort: pick out any dicts with 'phrase' and 'category' keys
        if not findings:
            findings = []
            for m in re.findall(r"\{[^\}]*\}", raw or ""):
                try:
                    d = json.loads(m)
                    if "phrase" in d and "category" in d:
                        findings.append(d)
                except Exception:
                    continue

        findings = [
            f for f in findings if isinstance(f, dict) and f.get("category") == "Abnormal"
        ]
        if not findings:
            return pd.DataFrame(columns=_FINDING_COLS)

        df = self._process_findings(findings, clinical_note)
        for col in _FINDING_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df

    def _process_findings(
        self,
        findings,
        clinical_note: str,
        metric: str = "cosine",
        keep_top: int = 15,
    ) -> pd.DataFrame:
        """
        Processes findings and returns DataFrame with phrase, category,
        metadata, sentence, patient_id.
        - keep_top: number of unique metadata entries to retrieve.
        """
        sentences = [s.strip() for s in clinical_note.split(".") if s.strip()]
        rows = []

        for f in findings:
            phrase = f.get("phrase", "").strip()
            category = f.get("category", "")
            if not phrase:
                continue

            qv = self._embed_query(phrase, metric=metric)

            unique_metadata = self._collect_metadata_best(
                phrase=phrase,
                query_vec=qv,
                index=self._index,
                docs=self._docs,
                top_k=500,
                similarity_threshold=0.35,
                min_unique=keep_top,
                max_unique=keep_top,
            )

            # Find the best-matching sentence by word overlap
            fw = set(_WORD_PATTERN.findall(phrase.lower()))
            best_sent, best_score = None, 0
            for s in sentences:
                score = len(fw & set(_WORD_PATTERN.findall(s.lower())))
                if score > best_score:
                    best_score, best_sent = score, s

            rows.append(
                {
                    "phrase": phrase,
                    "category": category,
                    "unique_metadata": unique_metadata,
                    "original_sentence": best_sent,
                    "patient_id": f.get("patient_id"),
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _collect_metadata_best(
        phrase: str,
        query_vec: np.ndarray,
        index: faiss.Index,
        docs: list[dict[str, Any]],
        top_k: int = 500,
        similarity_threshold: float = 0.35,
        min_unique: int = 15,
        max_unique: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Single-pass hybrid retrieval: token overlap, threshold, fill to min_unique.
        """
        clean_tokens = set(_WORD_PATTERN.findall(phrase.lower()))
        dists, idxs = index.search(query_vec, top_k)
        sims, indices = dists[0], idxs[0]

        seen_hp = set()
        results = []

        for sim, idx in sorted(zip(sims, indices), key=lambda x: x[0], reverse=True):
            if len(results) >= max_unique:
                break
            if idx < 0:  # FAISS pads with -1 when fewer than top_k vectors
                continue
            doc = docs[idx]
            hp = doc.get("hp_id")
            if not hp or hp in seen_hp:
                continue

            info = doc.get("info", "") or ""
            token_overlap = bool(clean_tokens & set(_WORD_PATTERN.findall(info.lower())))

            # accept if token overlap, or above similarity threshold, or to reach min_unique
            if token_overlap or sim >= similarity_threshold or len(results) < min_unique:
                seen_hp.add(hp)
                results.append(
                    {
                        "hp_id": hp,
                        "phrase": info,
                        "definition": doc.get("definition"),
                        "organ_system": doc.get("organ_system"),
                        "similarity": float(sim),
                    }
                )

        return results

    @staticmethod
    def _create_faiss_index(emb_matrix: np.ndarray, metric: str = "cosine"):
        # Build FAISS index for embeddings
        dim = emb_matrix.shape[1]
        if metric == "cosine":
            faiss.normalize_L2(emb_matrix)
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexFlatL2(dim)
        index.add(emb_matrix)
        return index

    @staticmethod
    def _load_vector_db(
        meta_path: Path, vec_path: Path
    ) -> tuple[list[HPODoc], NDArray[np.float32]]:
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{meta_path} not found. Compute embeddings first.")
        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"{vec_path} not found. Compute embeddings first.")

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                combined = json.load(f)
                constants = combined.get("constants", {})
                entries = combined.get("entries", [])
        except Exception as e:
            raise IOError(
                f"Could not load {meta_path}. Please, compute embeddings before using HPO RAG"
            ) from e

        try:
            arr = np.load(vec_path)
            emb_matrix = np.ascontiguousarray(arr["emb"].astype(np.float32))
        except Exception as e:
            raise IOError(
                f"Could not load {vec_path}. Please, compute embeddings before using HPO RAG"
            ) from e

        if len(entries) != emb_matrix.shape[0]:
            logger.warning(
                "Metadata entries count and embedding rows mismatch (%d vs %d)",
                len(entries),
                emb_matrix.shape[0],
            )

        docs = []
        for entry, vec in zip(entries, emb_matrix):
            hp_id = entry.get("hp_id")
            const = constants.get(hp_id, {})
            docs.append(
                {
                    "hp_id": hp_id,
                    "info": entry.get("info"),
                    "lineage": const.get("lineage"),
                    "organ_system": const.get("organ_system"),
                    "direction": entry.get("direction"),
                    "depth": const.get("depth"),
                    "parent_count": const.get("parent_count"),
                    "child_count": const.get("child_count"),
                    "descendant_count": const.get("descendant_count"),
                    "embedding": vec,
                }
            )

        return docs, emb_matrix