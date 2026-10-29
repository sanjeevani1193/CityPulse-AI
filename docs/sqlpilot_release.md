# SQLPilot: live baseline, diagnostic evidence and standalone Docker

## Measured live baseline

The first genuine Groq `openai/gpt-oss-20b` evaluation used the nine frozen
`tests/fixtures/sqlpilot_questions.json` cases against the approved small local
sample. All nine exercised real generation. It achieved **6/9 correct outcomes**:
4/6 answerable results correct, 5/6 answerable queries executed, 2/3 expected
clarification/safety statuses correct, and 1/1 malicious instruction declined.
Median agent latency was **1.946 seconds**; the paced batch took 522.018 seconds.
There was no provider stopping error. This is a tiny development evaluation,
not a general text-to-SQL accuracy or security guarantee.

The original ignored report remains unchanged at
`outputs/sqlpilot/groq-20b-live-evaluation.json`, SHA-256
`fede71da44031585ab1b1a6315849074a17708993ef2ccf24232ab875d8e5f00`.
The frozen question file's byte SHA-256 is
`0f5a5989d9b7cdbfa060ae7db3e3bc988011b27d01785b55517806006af62f1a`;
the baseline report records its canonical-JSON hash separately.

## Failure investigation: evidence and limits

| Case | Established evidence | What cannot yet be established |
| --- | --- | --- |
| `top_station` | One generated query executed; positional result comparison failed. | SQL, projected columns and actual/reference rows were not saved. Aggregation errors, tie ordering and projection differences remain possible; none is established. |
| `weekday` | Two generated queries failed at the query tool; the one-correction budget was exhausted. | No per-attempt SQL or error was retained. Policy rejection, binding, timezone syntax and execution timeout cannot be distinguished. This was not a provider-generation failure. |
| `ambiguous` | The model produced executable SQL instead of the required clarification. The graph correctly routed that SQL decision to execution. | The model's assumed ranking metric and returned results were not saved. The graph delegates semantic ambiguity detection to generation; safe SQL execution does not prove the question was well-defined. |

The baseline exporter discarded SQL, results and traces even though `run_agent`
returned them. **Retrospective exact diagnosis is blocked by missing evidence.**
No reference-specific SQL, weaker allowlists, extra repair attempts or automatic
live reruns were introduced. Supply original saved UI traces if available.

New evaluations retain per-attempt SQL, sanitized query-tool errors, actual and
reference results, agent traces and MCP outcomes **locally in the create-only
report**. Reference results never enter model prompts or correction feedback.
These reports can contain sample rows and generated SQL; keep them gitignored
and do not publish them unreviewed. This instrumentation is not evidence of
improved live accuracy. No second live batch has been run for this change.

After account/data-sharing approval is still confirmed, a single further batch
may be useful for diagnosis. Use a new filename (never overwrite the baseline),
existing 65-second spacing, unchanged frozen questions, and the same sample:

```sh
PYTHONPATH=src .venv-sqlpilot/bin/python -m citypulse_ai.sqlpilot.evaluate_graph \
  --database outputs/historical/samples/2024-01-n1000/citypulse.duckdb \
  --output outputs/sqlpilot/groq-20b-diagnostic-evaluation.json
```

Do not enable this until ready for actual external requests. Follow
[provider and privacy gates](sqlpilot_groq_setup.md). New results on this already
examined corpus would be a development follow-up, not an independent blind test.

## Architecture and security

Streamlit → LangGraph → official MCP SDK over stdio → internal `mcp_server` →
isolated read-only DuckDB workers. The container preserves this entire path and
the distinct internal module name. It copies only SQLPilot source, package
initializer, Streamlit configuration and pinned dependency definitions. It does
not contain forecasting modules, datasets, `.env` files or credentials. The
Dockerfile-specific allowlist controls the build context; Forecasting v1's
Dockerfiles, Compose and root `.dockerignore` are unchanged.

The Python 3.11 base uses the already verified digest from the worker image.
Application dependencies use existing pins; a separate `/opt/sqlpilot-core`
interpreter contains DuckDB/pytz only. No ML model, GPU runtime or local LLM is
installed. The image includes existing pytest for reproducible regression checks.
The existing core enforces 64 MB DuckDB memory, one thread, no spill/external
access, a five-second worker deadline, one allowlisted SELECT, 100 returned rows,
and 128 KiB result budget. Compose runs UID 10001, drops capabilities, prevents
privilege escalation, limits RAM to 768 MiB, one CPU and 64 PIDs, and makes the
root filesystem and sample mount read-only. Small ephemeral home/tmp mounts are
writable. These are bounds, not proof against arbitrary native-code exploits.

## Build, start, use and stop

An image packages the application and dependencies; a container is a running
instance. The Dockerfile builds the image; Compose declares its runtime, limits,
port and mounts. MCP uses child-process stdio within that one container, so no
MCP network port is exposed. A bind mount supplies the existing approved sample
without copying it into the image. A health check verifies Streamlit's HTTP
endpoint; it does not prove an LLM is configured or accurate.

On a machine with adequate resources (prefer the GitHub runner on an 8 GB Mac):

```sh
export SQLPILOT_SAMPLE_DATABASE="$PWD/outputs/historical/samples/2024-01-n1000/citypulse.duckdb"
docker compose -f compose.sqlpilot.yaml config --quiet
docker compose -f compose.sqlpilot.yaml build
docker compose -f compose.sqlpilot.yaml up -d --wait
# Open http://127.0.0.1:8501 and inspect the permitted schema.
docker compose -f compose.sqlpilot.yaml down
```

Only `127.0.0.1:8501` is published. Keep it local; public hosting/authentication
is outside this release. The sample must be a readable, trusted physical DuckDB
file, at most 32 MiB and 10,000 rows/table, not a symlink or historical view catalog.
A missing bind source fails rather than creating a directory. Never mount all
project outputs, credentials or historical data.

Provider generation is disabled unless all existing approval gates are set.
After explicit approval, obtain the key with the documented hidden shell prompt
and export the approved Groq variables. Compose forwards the key and approval
values at runtime only. Do not put secrets in build arguments, image layers or
Git. Avoid printing full `docker compose config` or `docker inspect` when secrets
are configured; `config --quiet` validates without displaying them. Docker
administrators can inspect runtime environment variables, so use a trusted host.
To revoke generation, unset the key/approval variables and recreate the service.

## Automated container verification and release status

The existing SQLPilot GitHub Actions workflow now has a separate standard
Ubuntu-hosted container job. It builds SQLPilot only, generates a tiny synthetic
DuckDB fixture, runs the full regression suite in an offline restricted
container, starts Compose, checks real HTTP health and WebSocket UI interaction,
clicks schema discovery through MCP, and executes a real LangGraph/MCP query
using an explicitly controlled generation double. It also checks clarification
routing and rejects destructive SQL. No provider secret is supplied and no LLM
accuracy is attributed to these checks. Runner-owned test containers are stopped
in an `always()` cleanup step. Image size, Docker stats/storage and host RAM/disk
are printed to reproducible Actions logs. A single usage sample is not a memory
safety guarantee.

Local verification: **65 tests passed** after adding diagnostic evidence capture;
workflow and Compose YAML parsed and whitespace checks passed.

**Hosted packaging verification passed** on revision `b6ac4e5`:
[SQLPilot CI run 40949596948](https://github.com/sanjeevani1193/CityPulse-AI/actions/runs/40949596948).
All three jobs succeeded: 42 core tests, 23 application/provider-preparation tests,
and all 65 tests again in the offline restricted image. The real HTTP/WebSocket
check rendered the UI and clicked the schema button; MCP negotiated `2025-03-26`,
discovered both tools and executed the LangGraph query. Clarification routing and
unsafe-SQL rejection passed. Generation in those hosted checks was controlled,
not live Groq inference. Compose reported healthy and its test container/network
were removed by the runner's cleanup step.

Measured image size: **806,594,684 bytes (769.23 MiB)**. The allowlisted application
build context transferred 49.42 kB. A post-check container sample was **92.96 MiB
of its 768 MiB limit**, 0.29% CPU and six PIDs. Runner RAM was 15 GiB total with
14 GiB available; swap use was zero. Docker reported 3.294 GB across seven images
(including pre-existing runner images), zero container writable-layer bytes and
49.85 kB reported build cache. These are observations from this run, not peak
memory, exclusive SQLPilot disk growth, or a guarantee for arbitrary queries.
No Docker build or container ran on the Mac.

The initial run for `38a633a` was rejected before jobs started. Its actual
[run annotation](https://github.com/sanjeevani1193/CityPulse-AI/actions/runs/40949520472)
reported `(Line: 58, Col: 33): Unrecognized named-value: 'runner'. Located at
position 1 within expression: runner.temp`. This was workflow expression
validation, not an absent trigger, insufficient permissions or test failure.
The sample path incorrectly referenced `runner.temp` at job-level `env`; commit
`b6ac4e5` moved it to `$RUNNER_TEMP` in a runner step. The corrected run above
passed all jobs without weakening resource or SQL safeguards. Documentation-only
changes now trigger SQLPilot checks too, and manual dispatch permits a fresh
verification on `main` without changing application code or enabling providers.

**Release scope:** standalone Docker packaging and the existing safe application
path are verified. Exact diagnosis of the two SQL failures and improved live
accuracy remain blocked by missing baseline evidence. Do not describe the three
baseline failures as fixed or call this an accuracy-certified SQLPilot release.

No QLoRA, ExperimentLab, historical ETL or Forecasting v1 changes are included.
