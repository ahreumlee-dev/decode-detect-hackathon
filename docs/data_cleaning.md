# Cleaning dependency

The notebook and synthetic regression tests come from Person 1's commit 0011134 (feature/data-cleaning), imported without datasets. Run `python notebooks/run_data_cleaning.py` from the repository root to execute every cell in a fresh kernel using the current Python interpreter. Install requirements-behaviour.txt first. See [the shared run guide](behaviour_features.md) for input paths, execution order and downstream model commands.

The source notebook has no saved outputs. Executed notebooks go to ignored outputs/. Cleaned train/validation/test tables and transfers go to ignored data/processed/. The helper supports `--notebook 04_behaviour_features.ipynb` for the same fresh-kernel verification of role 04.
