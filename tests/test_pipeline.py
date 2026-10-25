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
    print("Loaded ontology")
    ontology = Ontology(ontology)
    print("Initializing Embedding Model")
    embedding_model = SentenceTransformerModel()
    print("Initializing LLM Client")
    llm_client = LLMClient(timeout=5000)
    print("Initializing RAG HPO Client")
    rag_hpo = RAGHPO.inti_and_create_embeddings(
        ontology,
        embedding_model,
        llm_client,
        hp_meta_file=Path(
            "/Users/rouvenreuter/Documents/Projects/RAG-HPO/hp_meta.json"
        ),
        emb_file=Path("/Users/rouvenreuter/Documents/Projects/RAG-HPO/embed.npz"),
    )
    print("Initializing Extracting")
    res = rag_hpo.extract(
        [
            "During the interview the patient displayed marked autistic behaviors, including poor eye contact and rigid, repetitive speech, and also reported ongoing hallucinations that had worsened over the past month."
        ]
    )
    res.to_csv("/Users/rouvenreuter/Documents/Projects/RAG-HPO/results.csv")
    print(res)
