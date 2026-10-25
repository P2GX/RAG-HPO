# RAG-HPO

**Extract Human Phenotype Ontology (HPO) terms from clinical notes with a large language model and retrieval-augmented generation.**

`rag_hpo` is a pip-installable Python package version of
[RAG-HPO](https://www.medrxiv.org/content/10.1101/2024.12.01.24318253v1), the method by
Brandon T. Garcia and colleagues (Posey Lab, Baylor College of Medicine). The original
was a Jupyter notebook. This package turns it into a library you can import and call
from scripts or pipelines. It adds a pluggable embedding backend, an LLM client for any
OpenAI-compatible endpoint, and checkpointing so long runs can resume.

RAG-HPO is not an LLM itself. It drives an LLM that you provide (local or hosted). The
model finds phenotype phrases in free text, and a vector index of the HPO grounds each
phrase to a real HPO ID.

> [!WARNING]
> **Protecting patient information is your responsibility.** Notes are sent to whatever
> LLM endpoint you configure. Always de-identify data before processing it, and only use
> hosted APIs that your institution allows for the data in question (HIPAA, GDPR, IRB
> requirements). Checkpoint files contain note text and must be protected the same way.

---

## How it works

```
clinical note ──► LLM: extract phrases ──► keep "Abnormal" ──► FAISS: top-15 HPO candidates
                                                                        │
                       ┌────────────── local match? (bag-of-words / exact / fuzzy ≥ 80)
                       │ yes                                            │ no
                       ▼                                                ▼
                   HPO ID  ◄──────────────────────────────  LLM: pick one candidate
                                                            (local fallback if it fails)
```

1. **Extract**: the LLM reads each note and returns phenotype phrases, each tagged
   `Abnormal`, `Normal`, `Family History`, `Other` or `Suspected`. Only `Abnormal`
   phrases go on to the next steps.
2. **Retrieve**: each phrase is embedded and matched against a FAISS index of every HPO
   label and synonym. The 15 most similar distinct terms become its candidates.
3. **Match locally**: phrases that match a candidate closely enough get an HPO ID
   without another LLM call.
4. **Map with the LLM**: for the remaining phrases, the LLM picks the best candidate,
   using the sentence the phrase came from as context.
5. **Compile**: results are grouped per patient.

The [pipeline guide](docs/pipeline.md) covers each step in detail.

## You need
- **The HPO ontology** in OBO format (`hp.obo`), from the
  [HPO release page](https://hpo.jax.org/data/ontology) or
  `https://purl.obolibrary.org/obo/hp.obo`.
- **An LLM endpoint** that speaks the OpenAI chat-completions API. The default is a local
  [Ollama](https://ollama.com) server. Groq, OpenAI, vLLM and LM Studio also work.

## Quickstart

```python
from pathlib import Path

import pronto

from rag_hpo.embedding_model import SentenceTransformerModel
from rag_hpo.llm_client import LLMClient
from rag_hpo.ontology import Ontology
from rag_hpo.rag_hpo import RAGHPO
from rag_hpo.hit import Hit

embedding_model = SentenceTransformerModel()  # SapBERT-based biomedical model
llm = LLMClient()  # Ollama at localhost:11434, model qwen3.5:4b

# One-time: build the HPO vector database (repeat when HPO releases a new version)
ontology = Ontology(pronto.Ontology("hp.obo"))
rag = RAGHPO.inti_and_create_embeddings(
    ontology,
    embedding_model,
    llm,
    hp_meta_file=Path("hpo_meta.json"),
    emb_file=Path("hpo_embedded.npz"),
)

# Later runs: load the existing database
rag = RAGHPO(embedding_model, llm, "hpo_meta.json", "hpo_embedded.npz")

results: dict[str, list[Hit]] = rag.extract(
    [
        "The patient has recurrent seizures and global developmental delay. Hearing is normal."
    ]
)

print(results["0"].hpo_id)
```

To process a CSV of notes, pass a DataFrame with a `clinical_note` column and,
optionally, a `patient_id` column:

```python
import pandas as pd

notes = pd.read_csv("notes.csv")  # columns: patient_id, clinical_note
results = rag.extract(notes, checkpoint_dir="checkpoints")
```

### Using a hosted LLM (Groq)

```python
import os

llm = LLMClient.groq(api_key=os.environ["GROQ_API_KEY"])  # llama-3.3-70b-versatile
```

`LLMClient.groq()` applies Groq's free-tier limits: 30 requests per minute and 500,000
tokens per day. The [configuration guide](docs/configuration.md) covers OpenAI, LM Studio
and other endpoints.

## Output

`extract()` returns a `dict[str, list[Hit]]` where the key represents a patient.

```json
{
  "1": [
    {
      "phrase": "poor eye contact",
      "category": "Abnormal",
      "hpo_id": "HP:0000817"
    },
    {
      "phrase": "marked autistic behaviors",
      "category": "Abnormal",
      "hpo_id": "No Candidate Fit"
    }
  ]
}
```

`hpo_id` is an HPO ID such as `HP:0001250`, or the string `"No Candidate Fit"` if
nothing could be mapped. Patients with no abnormal findings are left out.

## Citation

If you use RAG-HPO in your work, please cite the original publication:

> Garcia BT, et al. *Improving Automated Deep Phenotyping Through Large Language Models
> Using Retrieval-Augmented Generation.* Genome Medicine (2025).
> [doi:10.1186/s13073-025-01521-w](https://doi.org/10.1186/s13073-025-01521-w)

## License

MIT. See [LICENSE](LICENSE). The original RAG-HPO is © 2024 Jennifer Posey Lab, Baylor
College of Medicine.