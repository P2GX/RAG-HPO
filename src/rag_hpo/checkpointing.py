import os
import shutil
from pathlib import Path

import pandas as pd


def save_state_checkpoint(
    state, temp_files, keys=("input", "combined", "exact", "non_exact", "final")
):
    # Saves pipeline state to disk
    for key in keys:
        df = state.get(key)
        if df is None or df.empty:
            continue
        rel_path = temp_files.get(key)
        if not rel_path:
            logger.log(f"Warning: No path configured for '{key}'. Skipping.")
            continue
        abs_path = os.path.abspath(rel_path)
        tmp_path = abs_path + ".tmp"
        try:
            df.to_pickle(tmp_path)
            shutil.move(tmp_path, abs_path)
            logger.log(
                f"[SAVE] Checkpointed '{key}' ({len(df)} rows) -> {abs_path}", once=True
            )
        except Exception as e:
            logger.log(f"Error saving '{key}': {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def load_state(temp_files: dict[str, Path]):
    # Loads pipeline state from temp files
    state = {}
    for key, path in temp_files.items():
        try:
            if os.path.exists(path):
                df = pd.read_pickle(path)
                state[key] = df
                # This print is now handled in main
            else:
                state[key] = pd.DataFrame()
        except Exception as e:
            state[key] = pd.DataFrame()
            print(f"Warning loading '{key}': {e}. Starting fresh for this key.")
    return state
