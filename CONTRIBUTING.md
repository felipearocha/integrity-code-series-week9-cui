# Contributing

Thanks for considering a contribution to the **ICS2 Week 9 — CUI** package.
This is a research and education repository (see [LICENSE](LICENSE)); contributions
should preserve **physics correctness, regulatory traceability, and reproducibility**
above all.

## Quick links

- [Project README](README.md)
- [Changelog](CHANGELOG.md)
- [Issue tracker](https://github.com/felipearocha/integrity-code-series-week9-cui/issues)
- [License](LICENSE)

## Ground rules

1. **Physics-first.** Any change to a governing equation, boundary condition,
   or material constant must reference the source (API code clause, peer-reviewed
   paper, or an `[ASSUMED]` flag with rationale). Update both the source code
   comment AND the README equation block.
2. **Tests stay green.** All 151+ tests must pass on Python 3.11, 3.12, and 3.13
   before a PR is mergeable. CI enforces this.
3. **Benchmarks stay tight.** The five analytical benchmarks in
   `validation/benchmarks.py` must close to within their existing tolerances
   (Faraday < 0.05 %, Arrhenius exact, BV→Tafel < 5 %, etc.).
4. **No silent baseline drift.** `run_all.py` `[1/9]` reports
   `wl(10yr) = 0.1360 mm` for the deterministic baseline. If your change moves
   that number, document the reason in `CHANGELOG.md` and explain it in the PR.
5. **Honest documentation.** If a feature is partial or has a constraint,
   say so in the README and the relevant docstring. The previous "DFOS inverse"
   bug (silent failure presented as success) is the kind of thing this rule exists
   to prevent.

## Development setup

```bash
git clone https://github.com/felipearocha/integrity-code-series-week9-cui.git
cd REPO
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                # installs runtime + dev tooling

# Sanity check
pytest tests/ -v                       # 151+ tests
python validation/benchmarks.py        # 5 analytical benchmarks
python run_all.py                      # full pipeline, ~70 s
```

## Tooling

| Tool        | Role                       | How to invoke                              |
|-------------|----------------------------|--------------------------------------------|
| pytest      | Unit + integration tests   | `pytest tests/`                            |
| ruff        | Lint + format              | `ruff check .` / `ruff format .`           |
| coverage    | Test coverage              | `pytest --cov=src`                         |
| pre-commit  | Pre-commit hooks           | `pre-commit install` then `pre-commit run --all-files` |

All are configured in `pyproject.toml` and `.pre-commit-config.yaml`.

## Branching and commits

- Branch from `main`. Use descriptive names: `fix/inverse-saturation-bound`,
  `feat/2d-axial-coupling`, `docs/changelog-1.2`.
- Conventional Commits encouraged but not enforced (e.g. `fix:`, `feat:`,
  `docs:`, `test:`, `refactor:`).
- Keep commits focused; avoid mixing physics changes with formatting churn.

## Pull requests

1. Open a draft PR early — it is fine to discuss approach before all tests pass.
2. CI must be green (all matrix cells) before review.
3. The PR description should answer:
   - What problem does this solve?
   - What did you change in physics, numerics, or API surface?
   - What tests / benchmarks did you add or modify?
   - Does this affect the `[ASSUMED]` parameter table or the baseline numbers?
4. Reviewers check physics traceability, test coverage, and CHANGELOG hygiene.

## Reporting issues

- **Bug**: use the **Bug report** issue template; include the failing command,
  full traceback, and `python --version` / OS.
- **Physics question**: use the **Physics question** template; cite the equation
  number, the API code clause if applicable, and the parameter values.
- **Feature request**: use the **Feature request** template; explain the
  industrial use case before the proposed API.

## Releasing (maintainers)

1. Bump version in `pyproject.toml`.
2. Update `CHANGELOG.md` `[Unreleased]` section into the new version.
3. Tag: `git tag -s vX.Y.Z -m "Release vX.Y.Z"`.
4. Push tag: `git push --tags`. The release workflow attaches CI artefacts.

## Security

If you discover a vulnerability that could mislead an integrity decision —
e.g. a sign error in a corrosion-rate path, a tamper-bypass in `audit_chain`,
or a numerical instability that produces silently wrong results — please
report it privately first via GitHub Security Advisories rather than a public issue.
