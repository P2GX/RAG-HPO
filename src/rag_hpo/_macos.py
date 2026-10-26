"""Work around duplicate OpenMP runtimes in macOS pip wheels (torch, faiss, sklearn).

Each wheel bundles its own libomp.dylib; running several multi-threaded
in one process hangs or segfaults.
"""
import importlib.util
import os
import sys


def apply() -> None:
    """Early defence: effective only if rag_hpo is imported before torch/faiss/sklearn."""
    if sys.platform != "darwin":
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if importlib.util.find_spec("torch") is not None:
        import torch  # noqa: F401


def limit_openmp() -> None:
    """Restrict all *loaded* OpenMP runtimes to one thread. Call after heavy imports."""
    if sys.platform != "darwin":
        return
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1, user_api="openmp")