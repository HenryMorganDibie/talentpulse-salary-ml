# Contributing

## Dev Setup

```bash
git clone https://github.com/HenryMorganDibie/talentpulse-salary-ml.git
cd talentpulse-salary-ml
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev]"
```

## Running the Pipeline

```bash
make run                           # default data path
make run-custom DATA=my_data.xlsx  # custom path
```

## Tests

```bash
make test        # full suite + coverage (must stay ≥ 80%)
make test-fast   # no coverage, faster iteration
```

All tests must pass before opening a PR. Coverage must not drop below 80%.

## Linting

```bash
make lint        # ruff check
make lint-fix    # auto-fix where possible
```

The CI pipeline runs `ruff` on every push. Fix lint errors locally before pushing.

## Branching

| Branch | Purpose |
|--------|---------|
| `main` | Stable, tagged releases only |
| `dev`  | Integration branch for new features |
| `feature/<name>` | Individual feature work |
| `fix/<name>`     | Bug fixes |

Open PRs against `dev`. `dev` → `main` is merged only at release time.

## Adding a New Model

1. Add the untrained estimator to `build_model_registry()` in `src/models/train.py`
2. Add its hyperparameter grid if applicable
3. Add it to `configs/pipeline.yaml` under a new section
4. Re-run the pipeline and update `reports/benchmarking_memo.md` with results
5. Update `CHANGELOG.md` under `[Unreleased]`

## Adding a New Feature

1. Add the transform function to `src/features/engineering.py`
2. Call it inside `build_features()`
3. Verify it appears in `get_feature_columns()` output
4. Add unit tests — at minimum: correctness, edge cases (NaN, empty list), and
   a leakage guard assertion
5. Confirm `test_no_salary_leakage_in_feature_cols` still passes

## Key Invariants — Do Not Break

- `salary_min` and `salary_max` must never appear in `get_feature_columns()` output
- `train_test_split_stratified` must not call `reset_index` — downstream alignment
  in `build_residuals_frame` depends on original DataFrame indices being preserved
- `score_model` must return only JSON-serialisable types (no numpy arrays)
- All plot functions must accept `out_dir: Path` explicitly — no module-level
  CWD-relative path defaults
