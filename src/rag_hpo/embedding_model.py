from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # imported only for type checkers, not at runtime
    from fastembed import TextEmbedding
    from sentence_transformers import SentenceTransformer

Embeddings = NDArray[np.float32]
ModelBackend = Literal["sentence-transformers", "fastembed"]


class EmbeddingModel(ABC):
    """Common interface: text in, normalized float32 matrix out."""

    model_name: str

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def encode(self, texts: Sequence[str], batch_size: int = 64) -> Embeddings:
        """Embed documents. Returns shape (n, dim), L2-normalized."""

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> Embeddings:
        """Embed search queries. Same as encode() unless a model needs otherwise."""
        return self.encode(texts, batch_size)

    @property
    @abstractmethod
    def dim(self) -> int: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model_name!r})"


class SentenceTransformerModel(EmbeddingModel):
    _model: SentenceTransformer

    def __init__(
        self,
        model_name: str = "pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb",
        device: str | None = None,
    ) -> None:
        super().__init__(model_name)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> Embeddings:
        vecs: np.ndarray = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32, copy=False)

    @property
    def dim(self) -> int:
        d: int | None = self._model.get_embedding_dimension()
        if d is None:
            raise RuntimeError(
                f"Could not determine embedding dimension for {self.model_name!r}"
            )
        return d


class FastEmbedModel(EmbeddingModel):
    _model: TextEmbedding
    _dim: int | None

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        super().__init__(model_name)
        from fastembed import TextEmbedding  # lazy import

        self._model = TextEmbedding(model_name=model_name)
        self._dim = None

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> Embeddings:
        vecs: Iterable[np.ndarray] = self._model.embed(
            list(texts), batch_size=batch_size
        )
        return self._stack(vecs)

    def encode_queries(self, texts: Sequence[str], batch_size: int = 64) -> Embeddings:
        # Applies the model's query prefix (matters for BGE retrieval)
        vecs: Iterable[np.ndarray] = self._model.query_embed(
            list(texts), batch_size=batch_size
        )
        return self._stack(vecs)

    def _stack(self, vecs: Iterable[np.ndarray]) -> Embeddings:
        arr: Embeddings = np.asarray(list(vecs), dtype=np.float32)
        if arr.size and self._dim is None:
            self._dim = int(arr.shape[1])
        return arr

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode(["probe"]).shape[1])
        return self._dim


_BACKENDS: dict[ModelBackend, type[EmbeddingModel]] = {
    "sentence-transformers": SentenceTransformerModel,
    "fastembed": FastEmbedModel,
}


def load_embedding_model(
    model_name: str, backend: ModelBackend, **kwargs: Any
) -> EmbeddingModel:
    try:
        cls: type[EmbeddingModel] = _BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"Unknown package {backend!r}; choose from {list(_BACKENDS)}"
        ) from None
    return cls(model_name, **kwargs)
