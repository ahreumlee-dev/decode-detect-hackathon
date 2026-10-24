# AI use disclosure

Record actual provider, model, prompts, relevant settings, and use case during the event.

## 2026-10-24 — cleaning notebook repair

- Provider/tool: OpenAI Codex. Exact model identifier and sampling settings were not exposed in the session.
- User prompt: repair `notebooks/01_data_cleaning.ipynb` on `feature/data-cleaning` for fresh-kernel reproducibility; restore the complete saved function, fix the weekend typo, remove session-history/source patching, preserve cleaning features, enforce exact issuer/hour USD prices, export all three splits and transfers, validate coverage/leakage, document the environment, and do not push or stage generated data.
- Assistance: inspected local changes, Git history, existing Python environment and organiser schema documentation; reorganised the notebook, added strict validation and a new-kernel runner, pinned cleaning dependencies, added synthetic regression checks, and executed/verified all outputs.
- Scope: cleaning only; no model training or change to the team's 80/20 model split. Raw data and existing unrelated user changes were preserved. No external labels or services were used.
- Reproduction commands and measured results: `docs/data_cleaning.md`.

- Follow-up authorization: the user approved committing and pushing the repair and three regenerated cleaned Parquet files to `feature/data-cleaning` so teammates can receive them. Added team fetch/merge, clone and working-directory guidance to the README; the kernel runner selects TCP on Windows and IPC on Unix.

## 2026-10-24 — initial ML baseline

- Provider/tool: OpenAI Codex; exact model identifier and sampling settings were not exposed.
- User prompt: continue `02_ml_baseline.ipynb`, using the provided team parameters (700 estimators, 0.02 learning rate, 63 leaves, etc.) as the starting configuration. The user supplied a GeeksforGeeks overview and confirmed the initial parameters.
- Assistance: retained and extended the loading cell; added an explicit feature list, train-only categorical vocabulary, the existing stratified 80/20 split, LightGBM training, AP/threshold/ranking evaluation, artifact exports/reload checks, pinned dependencies and a fresh-kernel runner. Consulted official LightGBM documentation and the installed package API.
- No tuning, official validation/test use, model submission or changes to the team's split. No final refit or test prediction generation. The notebook was executed from top to bottom in a new kernel; measured results are in `docs/ml_baseline.md`.

## 2026-10-24 — prepare graph/behaviour handoff

- User clarified the objective: prepare the ML code now so Person 2 and Person 4 feature deliveries can be tested later; the current baseline score is only a reference. The role document was treated as contextual team requirements.
- Codex added optional configured feature inputs, strict label-free numeric joins on transaction hashes, prefixed columns, experiment-isolated artifacts, matching-input/split/parameter baseline comparisons and ten synthetic regression tests. The split and original LightGBM parameters remain unchanged. No graph/behaviour feature values were invented. RF/eight-model grid and final submission remain future work.
- Re-executed the actual baseline with no optional files in a fresh kernel. Official validation/test were not read. No push was made for the ML work.

## 2026-10-24 — test Thanmaya's Person 4 contribution

User requested retrieving and testing Thanmaya's uploaded work. Codex fetched repository refs, identified `behaviour-features-pr-04` at `1935686`, inspected the notebook and imported only its behaviour implementation. Local scope/output adaptations generate train only and retain our current seed-42 split. No feature calculations were changed. Allowed intentional behaviour NaN gaps for native LightGBM handling, added two regression tests, generated features in a new kernel and ran the enriched ML notebook in another new kernel. Saved exact split equality, provenance and AP comparison in `docs/thanmaya_behaviour_comparison.md`. No graph features, official test scoring, hyperparameter changes, commit or push.

## 2026-10-24 — explicit training/holdout accuracy

At the user's request, added an accuracy table for both internal training and holdout at the fixed 0.5 threshold, alongside the always-normal reference and the official AP metric. Values are calculated from the same selected model, also saved to `accuracy_summary.csv` and metrics JSON. Re-executed the notebook in a fresh kernel and retained its verified outputs in the source notebook for immediate viewing. Training accuracy 97.7417%; internal holdout accuracy 97.2842%. No split, feature or model parameter changes; official test not read.

## 2026-10-24 — ML publication authorization

The user explicitly requested pushing the verified ML experiment. Publication includes the ML notebook with explanatory cells and measured outputs, the train-only adapted Person 4 notebook with upstream attribution, execution helpers, the strict feature adapter and tests, pinned requirements and reproduction/comparison documentation. Generated features, model files and predictions under `outputs/`, raw data and unrelated user changes are excluded.
