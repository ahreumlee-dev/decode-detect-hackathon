# AI use disclosure

Record actual provider, model, prompts, relevant settings, and use case during the event.

## 2026-10-24 — cleaning notebook repair

- Provider/tool: OpenAI Codex. Exact model identifier and sampling settings were not exposed in the session.
- User prompt: repair `notebooks/01_data_cleaning.ipynb` on `feature/data-cleaning` for fresh-kernel reproducibility; restore the complete saved function, fix the weekend typo, remove session-history/source patching, preserve cleaning features, enforce exact issuer/hour USD prices, export all three splits and transfers, validate coverage/leakage, document the environment, and do not push or stage generated data.
- Assistance: inspected local changes, Git history, existing Python environment and organiser schema documentation; reorganised the notebook, added strict validation and a new-kernel runner, pinned cleaning dependencies, added synthetic regression checks, and executed/verified all outputs.
- Scope: cleaning only; no model training or change to the team's 80/20 model split. Raw data and existing unrelated user changes were preserved. No external labels or services were used.
- Reproduction commands and measured results: `docs/data_cleaning.md`.

- Follow-up authorization: the user approved committing and pushing the repair and three regenerated cleaned Parquet files to `feature/data-cleaning` so teammates can receive them. Added team fetch/merge, clone and working-directory guidance to the README; the kernel runner selects TCP on Windows and IPC on Unix.
