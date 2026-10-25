from dataclasses import dataclass


@dataclass
class Hit:
    phrase: str
    category: str
    hpo_id: str
