from typing import TypedDict, Any

import numpy as np
from numpy.typing import NDArray


class HPODoc(TypedDict):
    hp_id: str
    info: str
    direction: int
    lineage: Any  # tighten once you know the stored format
    organ_system: Any
    depth: int | None
    parent_count: int | None
    child_count: int | None
    descendant_count: int | None
    embedding: NDArray[np.float32]
