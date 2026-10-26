import json
from dataclasses import asdict
from pathlib import Path

import pronto


from rag_hpo.embedding_model import SentenceTransformerModel
from rag_hpo.llm_client import LLMClient
from rag_hpo.ontology import Ontology
from rag_hpo.rag_hpo import RAGHPO


def test_pipeline():
    ontology = pronto.Ontology(
        str("/Users/rouvenreuter/Downloads/hp (4).obo"), encoding="utf-8"
    )

    ontology = Ontology(ontology)

    embedding_model = SentenceTransformerModel()

    llm_client = LLMClient(timeout=5000)

    rag_hpo = RAGHPO.inti_and_create_embeddings(
        ontology,
        embedding_model,
        llm_client,
        hp_meta_file=Path(
            "/Users/rouvenreuter/Documents/Projects/RAG-HPO/hp_meta.json"
        ),
        emb_file=Path("/Users/rouvenreuter/Documents/Projects/RAG-HPO/embed.npz"),
    )

    res = rag_hpo.extract(
        [
            "During the interview the patient displayed marked autistic behaviors, including poor eye contact and rigid, repetitive speech, and also reported ongoing hallucinations that had worsened over the past month."
        ]
    )

    with open("/Users/rouvenreuter/Documents/Projects/RAG-HPO/hits.json", "w") as f:
        json.dump({k: [asdict(h) for h in v] for k, v in res.items()}, f, indent=2)
