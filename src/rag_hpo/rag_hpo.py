import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any
from rapidfuzz import fuzz as rfuzz
import pandas as pd

from rag_hpo.llm_client import LLMClient
from utils import _WHITESPACE_PATTERN, _WORD_PATTERN, _JSON_OBJECT_PATTERN, _clean_text
from utils import _NON_ASCII_PATTERN
import faiss
import numpy as np
from numpy._typing import NDArray

from rag_hpo.embedder import Embedder
from rag_hpo.embedding_model import EmbeddingModel
from rag_hpo.hpo_df_builder import HPODataframeBuilder
from rag_hpo.hpo_doc import HPODoc
from rag_hpo.ontology import Ontology

import logging

logger = logging.getLogger(__name__)


class RAGHPO:
    def __init__(self, embedding_model: EmbeddingModel, llm_client: LLMClient):
        self._embedding_model = embedding_model
        self._llm_client = llm_client
        pass

    @classmethod
    def inti_and_create_embeddings(
        cls,
        ontology: Ontology,
        embedding_model: EmbeddingModel,
        hp_meta_file: Path,
        emb_file: Path,
    ):
        df_builder = HPODataframeBuilder(ontology)
        hpo_df = df_builder.build_hpo_dataframe()
        embedder = Embedder(embedding_model)
        embedder.get_embedding(hpo_df, hp_meta_file, emb_file)

    def extract(self):
        pass

    def _embed_query(self, text: str, metric: str = "cosine"):
        vec = self._embedding_model.encode(text)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        if metric == "cosine":
            faiss.normalize_L2(vec)
        return vec

    @staticmethod
    def _clean_note(text: str) -> str:
        text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        text = unicodedata.normalize("NFKD", text)
        text = _WHITESPACE_PATTERN.sub(text)
        text = _NON_ASCII_PATTERN.sub(text).strip()
        return text

    @staticmethod
    def _split_exact_nonexact(df: pd.DataFrame, hpo_term_col="HPO_Term"):
        # print(f">> DEBUG: about to split, columns: {df.columns.tolist()}")
        if "category" not in df.columns:
            raise KeyError(
                f"[FATAL] 'category' missing; columns: {df.columns.tolist()}"
            )
        if hpo_term_col not in df.columns:
            raise KeyError(
                f"[FATAL] '{hpo_term_col}' missing; columns: {df.columns.tolist()}"
            )

        # only keep Abnormal rows
        df_ab = df[df["category"] == "Abnormal"]
        # print(f">> DEBUG: {len(df_ab)} rows with category=='Abnormal'")

        exact_df = df_ab.dropna(subset=[hpo_term_col]).copy()
        non_exact_df = df_ab[df_ab[hpo_term_col].isna()].copy()
        # print(f">> DEBUG: exact_df.shape={exact_df.shape}, non_exact_df.shape={non_exact_df.shape}")
        return exact_df, non_exact_df

    @staticmethod
    def _clean_and_parse(s: str):
        # Extracts and parses JSON from string
        try:
            m = _JSON_OBJECT_PATTERN.search(s, flags=re.S)
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
    def cleanup(temp_files, success):
        # Removes temp files if pipeline succeeded
        if success:
            logger.log("Pipeline succeeded. Cleaning up temporary files...")
            for path in temp_files.values():
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        # logger.log(f"Removed temp file: {path}")
                except OSError as e:
                    logger.log(f"Error removing temp file {path}: {e}")
        else:
            logger.log("Pipeline failed. Keeping temporary files for debugging/resume.")

    @staticmethod
    def _process_results(final_df: pd.DataFrame):
        # Handles output of final results (CSV or display)
        if final_df.empty:
            logger.log("No final results to process.")
            return
        choice = (
            input("Save results as CSV or display? (save/display): ").strip().lower()
        )
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
                            f"Warning: Blank HPO_Term for patient {pid}, phrase '{ph}', category '{cat}' - not included in CSV."
                        )
            if rows:
                output_df = pd.DataFrame(rows)
                output_df.to_csv(fname, index=False)
                logger.log(f"Saved tabular results to {fname}")
            else:
                logger.log("No valid HPO terms to save in tabular format.")
            final_df.to_csv(f"{os.path.splitext(fname)[0]}_json_raw.csv", index=False)
            logger.log(
                f"Saved raw JSON results to {os.path.splitext(fname)[0]}_json_raw.csv"
            )
        elif choice == "display":
            tbl = []
            for idx, r in final_df.iterrows():
                pid = r.get("patient_id", idx)
                for term in r.get("HPO_Terms", []):
                    ph = term.get("phrase", "").strip()
                    cat = term.get("category", "")
                    hp = term.get("HPO_Term") or ""
                    if isinstance(hp, str):
                        hp = hp.replace("HP:HP:", "HP:")
                    tbl.append(
                        {
                            "Case": f"Case {pid}",
                            "Category": cat,
                            "Phenotype name": ph,
                            "HPO ID": hp,
                        }
                    )
            if tbl:
                print(tabulate(pd.DataFrame(tbl), headers="keys", tablefmt="psql"))
            else:
                logger.log("No terms to display.")
        else:
            print("Invalid choice; please enter 'save' or 'display'.")

    @staticmethod
    def _validate_input(df: pd.DataFrame):
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
    def _parse_llm_mapping(resp_text: str, candidate_ids: set) -> tuple[str|None, str, dict|None]:
        """
        Simplified parsing: strict JSON → key lookup → regex fallback.
        """
        # 1. Strict JSON parse
        try:
            js = json.loads(resp_text)
        except json.JSONDecodeError:
            js = None

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
    def _extract_hpo_term(phrase: str, metadata_list: list, cluster_index: defaultdict) -> str | None:
        # Maps phenotype phrase to best HPO term using multiple strategies
        if not metadata_list or (
            isinstance(metadata_list, float) and pd.isna(metadata_list)
        ):
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
            score = rfuzz.token_sort_ratio(cp, ct)
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
            term_text = d.get("info") or d.get("label")
            hp = d["hp_id"]
            if hp and term_text:
                yield term_text, hp
            return
        for k, v in d.items():
            if isinstance(v, str) and v.startswith("HP:"):
                yield k, v


    def _process_row(self,
        clinical_note, system_message, embeddings_model, index, embedded_documents
    ):
        # Clean the clinical note before sending to the LLM
        clinical_note = RAGHPO._clean_note(clinical_note)
        # Query the LLM
        raw = self._llm_client.query(clinical_note, system_message)

        # Try to parse findings robustly
        findings = self._extract_findings(raw)
        # If findings is empty, try to parse as a list of dicts (sometimes LLMs return just a list)
        if not findings:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    findings = parsed
            except Exception:
                pass

        # If still empty, try to extract any dicts with 'phrase' and 'category' keys
        if not findings:
            try:
                matches = re.findall(r"\{[^\}]*\}", raw)
                findings = []
                for m in matches:
                    try:
                        d = json.loads(m)
                        if "phrase" in d and "category" in d:
                            findings.append(d)
                    except Exception:
                        continue
            except Exception:
                pass

        findings = [
            f
            for f in findings
            if isinstance(f, dict) and f.get("category") == "Abnormal"
        ]

        # Return empty DataFrame if no valid findings, but with required columns
        required_cols = [
            "phrase",
            "category",
            "unique_metadata",
            "original_sentence",
            "patient_id",
        ]
        if not findings:
            return pd.DataFrame(columns=required_cols)
        # Continue processing
        df = self._process_findings(
            findings, clinical_note, embeddings_model, index, embedded_documents
        )
        # Ensure all required columns are present
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        return df

    @staticmethod
    def _process_findings(
        findings,
        clinical_note: str,
        embeddings_model,
        index,
        docs,
        metric: str = "cosine",
        keep_top: int = 15,
    ):
        """
        Processes findings and returns DataFrame with phrase, category,
        metadata, sentence, patient_id.
        - keep_top: number of unique metadata entries to retrieve.
        """
        # ─── Position A: Split note into sentences for context matching ───
        sentences = [s.strip() for s in clinical_note.split(".") if s.strip()]
        rows = []

        for f in findings:
            phrase = f.get("phrase", "").strip()
            category = f.get("category", "")
            if not phrase:
                continue

            # ─── Position B: Embed the phrase ───
            qv = RAGHPO._embed_query(phrase, embeddings_model, metric=metric)

            # ─── Position C: Retrieve best metadata candidates ───
            unique_metadata = RAGHPO._collect_metadata_best(
                phrase=phrase,  # literal text for token-phase
                query_vec=qv,  # embedded vector
                index=index,  # FAISS index
                docs=docs,  # list of HPO docs
                top_k=500,  # FAISS retrieval size
                similarity_threshold=0.35,  # max distance for semantic matches
                min_unique=keep_top,  # ensure at least keep_top entries
                max_unique=keep_top,  # cap at keep_top entries
            )

            # ─── Position D: Find the best-matching sentence ───
            fw = set(_WORD_PATTERN.findall(phrase.lower()))
            best_sent, best_score = None, 0
            for s in sentences:
                sw = set(_WORD_PATTERN.findall(s.lower()))
                score = len(fw & sw)
                if score > best_score:
                    best_score, best_sent = score, s

            # ─── Position E: Collect row ───
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
        Single‐pass hybrid retrieval: token overlap, threshold, fill to min_unique.
        """
        clean_tokens = set(_WORD_PATTERN.findall(phrase.lower()))
        dists, idxs = index.search(query_vec, top_k)
        sims, indices = dists[0], idxs[0]

        seen_hp = set()
        results = []

        for sim, idx in sorted(zip(sims, indices), key=lambda x: x[0], reverse=True):
            if len(results) >= max_unique:
                break
            doc = docs[idx]
            hp = doc.get("hp_id")
            if not hp or hp in seen_hp:
                continue

            info = doc.get("info", "") or ""
            token_overlap = bool(
                clean_tokens & set(_WORD_PATTERN.findall(info.lower()))
            )

            # accept if token overlap, or above similarity threshold, or to reach min_unique
            if (
                token_overlap
                or sim >= similarity_threshold
                or len(results) < min_unique
            ):
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

    def _load_vector_db(
        self, meta_path: Path, vec_path: Path
    ) -> tuple[list[HPODoc], NDArray[np.float32]]:
        if not os.path.exists(meta_path):
            raise FileNotFoundError()
        if not os.path.exists(vec_path):
            raise FileNotFoundError()

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
            emb_matrix = arr["emb"].astype(np.float32)
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

        # ─── Reconstruct docs list in the original output format ───
        docs = []
        for entry, vec in zip(entries, emb_matrix):
            hp_id = entry.get("hp_id")
            const = constants.get(hp_id, {})

            doc = {
                "hp_id": hp_id,
                "info": entry.get("info"),
                "lineage": const.get("lineage"),
                "organ_system": const.get("organ_system"),
                "direction": entry.get("direction"),
                # preserve these keys even if absent in the new JSON:
                "depth": const.get("depth"),
                "parent_count": const.get("parent_count"),
                "child_count": const.get("child_count"),
                "descendant_count": const.get("descendant_count"),
                "embedding": vec,
            }
            docs.append(doc)

        return docs, emb_matrix
