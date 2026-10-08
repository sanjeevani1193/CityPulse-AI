# Guidance for coding agents

- Explain key design decisions, tradeoffs, and verification results in plain language.
- Prefer simple, readable solutions. Add abstractions and dependencies only when a concrete requirement needs them.
- Keep application code in `src/citypulse_ai/` and tests in `tests/`.
- Write meaningful tests for new behavior and bug fixes; run the relevant tests and report failures or missing dependencies honestly.
- Never invent benchmark results, model scores, experiment outcomes, or successful test runs. Report only measured results and describe the data and evaluation method.
- Prevent temporal leakage when implementing features, train/test splits, and evaluation. Document time zones, aggregation grains, and missing-data policies.
- Implement only the requested phase. Do not download datasets, provision infrastructure, commit, or push without user authorization.
- Preserve existing Git configuration and unrelated user changes. Keep credentials and large datasets out of source control.
