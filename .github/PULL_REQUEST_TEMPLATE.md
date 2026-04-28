## Summary

<!-- 1-3 sentences. What problem does this solve? -->

## Changes

- Physics / equations:
- Numerics / solvers:
- API surface:
- Tests / benchmarks:

## Validation

- [ ] `pytest tests/ -v` — all 151+ tests pass locally
- [ ] `python validation/benchmarks.py` — all 5 benchmarks pass
- [ ] `python run_all.py` — pipeline completes; figures regenerate cleanly
- [ ] `ruff check .` and `ruff format --check .` clean
- [ ] `pre-commit run --all-files` clean

## Baseline check

- [ ] `wl(10yr) = 0.1360 mm` for the deterministic baseline (unchanged)
- [ ] If changed, the new value is justified and documented in CHANGELOG

## Documentation

- [ ] Updated README equations / table if the change is physics-visible
- [ ] Updated `[ASSUMED]` parameter table if a new constant was introduced
- [ ] Added/updated CHANGELOG entry under the Unreleased section
- [ ] Updated docstrings on touched public functions

## Linked issues

<!-- Closes #123 / Refs #456 -->

## Notes for reviewers

<!-- Anything non-obvious about the approach, alternatives considered,
     limits of the implementation, follow-up work. -->
