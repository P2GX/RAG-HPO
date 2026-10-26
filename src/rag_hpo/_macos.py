"""Work around duplicate OpenMP runtimes in macOS pip wheels (torch, faiss, sklearn).

Each wheel bundles its own libomp.dylib; with several loaded in one process,
multi-threaded OpenMP code hangs or segfaults. Restricting OpenMP to one
thread and loading torch's copy first avoids it.
Must run before torch, faiss, sklearn or numpy-heavy code is imported.
"""
import importlib.util
import os
import sys


def apply() -> None:
    if sys.platform != "darwin":
        return
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if importlib.util.find_spec("torch") is not None:
        import torch  # noqa: F401