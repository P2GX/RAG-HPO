from importlib.resources import files

import pandas as pd


def test_data() -> pd.DataFrame:
    """Load the bundled clinical-notes example dataset.

    Returns a DataFrame with columns ``patient_id`` (int) and ``clinical_note`` (str).
    """
    resource = files("rag_hpo.data").joinpath("clinical_notes.csv")
    with resource.open("r", encoding="utf-8") as f:
        return pd.read_csv(f, dtype={"patient_id": "int64", "clinical_note": "string"})
