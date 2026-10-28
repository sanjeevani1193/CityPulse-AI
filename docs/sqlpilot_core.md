# SQLPilot: deterministic query and MCP core

The later [standalone application milestone](sqlpilot_application.md) adds real
LangGraph orchestration, the official MCP client and Streamlit. This document
records the unchanged core and its earlier verification; live LLM approval is
still pending.

SQLPilot is a separate component under `src/citypulse_ai/sqlpilot/`. Forecasting v1
models, datasets, services, checkpoints and evaluation are unchanged. This milestone
implements the safe analytical substrate; **SQLPilot v1 text-to-SQL is not complete**.
No real LLM has been exercised and no natural-language accuracy is claimed.

## What existed and what is reused

The prior `sqlpilot` package contained only a placeholder `__init__.py`. There was
no text-to-SQL implementation, LangChain, LangGraph, MCP SDK, SQLGlot or configured
LLM API credentials in the inspected environment. DuckDB 1.5.6 is already installed.
The Ollama executable and a cached Gemma 3 manifest exist (approximately 3.34 GB of
model layers), but its loopback API is unreachable. macOS reports warning-level
memory pressure and approximately 6.64 GiB swap use at inspection; no model/server
was started and no model was downloaded. This does not establish that the cached
model is corrupt, nor that loading it would be safe.

The bounded existing sample database is:
`outputs/historical/samples/2024-01-n1000/citypulse.duckdb`.
Despite its nominal sample-directory name, inspection found **1,998 accepted trips**
and **1,931 station-hour rows**. It is not the final 100-station or complete two-year
dataset. The source database also has a February view; SQLPilot does not expose or
bind that view. No source paths, ride IDs or other hidden columns are exposed.
No full historical data, GCS restoration, new ETL or infrastructure is required.

## Architecture

`SQLPilot.ask(question)` → approved backend decision (SQL or clarification) →
DuckDB native AST validation → isolated readonly query worker → structured rows and
an explanation grounded in actual row count and source scope.

The backend interface is injectable but **no real provider adapter is configured**.
Without one, questions return `backend_unavailable`; they never trigger a guessed
query, paid API, local model launch or invented answer. Tests use explicit doubles
to verify clarification and unsafe-generation handling; these are not LLM results.
Schema is passed to a future backend, not arbitrary sample rows. Returned answers
currently explain the executed result, not a model-generated narrative.

### Permitted schema

| Physical table | Permitted columns | Grain |
| --- | --- | --- |
| `hourly_rides` | `start_station_id`, `hour_start`, `ride_count` | Accepted sampled starts per station/elapsed hour |
| `validated_trips` | `started_at`, `ended_at`, `start_station_id`, `end_station_id`, `duration_seconds` | One accepted sampled trip |

Station identifiers remain strings. Connections use UTC; NYC weekday/hour queries
explicitly use `AT TIME ZONE 'America/New_York'`. Repeated DST hours are distinct
UTC instants; grouping by local hour may intentionally combine them. Missing sample
rows do not establish zero physical demand, station operation or complete source
coverage. The core does not fill missing station-hours or claim historical totals.

## Safety boundary

- DuckDB's **native** `json_serialize_sql` parser, not regular expressions, validates
  one plain SELECT. Canonical SQL is generated with `json_deserialize_sql`.
- Pin DuckDB **1.5.6** for this AST contract; unknown versions fail closed. The
  optional `sqlpilot` dependency group records this pin without installing anything.
- Permit only two physical `main` tables and named columns. Deny views, catalogs,
  table functions, external sources, stars, CTEs, subqueries and set operations.
- Explicit function allowlist: count/sum/avg/min/max, round/abs, date_part/date_trunc,
  timezone/strftime, coalesce/lower/upper. Other functions fail closed. Reject
  non-builtin macro shadowing, hidden-column aliases and table-column renaming.
- Joins require explicit conditions. SELECT lists have at most 16 expressions;
  input SQL is at most 4 KiB. This intentionally supports a small SQL subset.
- Each query opens the operator-selected database **read_only=True** in a fresh
  child process. The tool caller cannot change its path or permissions.
- External access, extension autoload/install and temporary spill are disabled;
  DuckDB memory is capped at 64 MB with one thread. Database files must be existing,
  non-symlink trusted samples of at most 32 MiB and at most 10,000 rows/table.
- Parent wall timeout defaults to 5 seconds (maximum configurable 10); the POSIX
  worker also has an independent 5-second hard deadline. Timeout/error cleanup
  kills and reaps the child. macOS/Linux are the supported runtime platforms.
- Results are capped at 100 rows with an explicit truncation indicator and a
  128 KiB serialized response cap. Binding/execution failures return a safe error.

This is defense in depth for a **trusted operator-owned database and restricted
local tools**, not a general SQL sandbox against DuckDB native-code vulnerabilities
or malicious database binaries. A database memory limit is not an OS RSS cap; the
Python/native worker has additional overhead. Broader SQL grammar or arbitrary
user-provided databases require separate security review. Do not expose this core
as an unauthenticated public service.

## Commands and actual example

Use the existing environment; no installation was performed for this milestone:

```sh
export PYTHONPATH=src
.venv/bin/python -m citypulse_ai.sqlpilot \
  --database outputs/historical/samples/2024-01-n1000/citypulse.duckdb --schema
.venv/bin/python -m citypulse_ai.sqlpilot \
  --database outputs/historical/samples/2024-01-n1000/citypulse.duckdb \
  --sql 'SELECT sum(ride_count) AS sample_starts FROM hourly_rides'
```

Measured output: columns `["sample_starts"]`, rows `[[1998]]`, `truncated=false`.
This is a **real readonly SQL execution example, not an LLM-generated query**.
Questions such as “Which three sample stations have the most starts?”, “What is
the average sampled trip duration?” or “How many sampled starts occur on each
NYC weekday?” are represented in the deterministic reference-query evaluation.
Joins use exact station ID plus UTC elapsed hour. Natural-language versions do not
execute automatically until an approved backend is available.

```sh
.venv/bin/python -m citypulse_ai.sqlpilot \
  --database outputs/historical/samples/2024-01-n1000/citypulse.duckdb \
  --question 'Which three stations have the most recorded starts in this sample?'
# Returns status=backend_unavailable; no SQL/answer is invented.
```

The database path is operator configuration. Do not substitute large production
catalogs with external-file views. Fresh checkouts can run all tests with tiny
synthetic fixtures; the existing local sample is gitignored and not redistributed.

## MCP-compatible tools

```sh
PYTHONPATH=src .venv/bin/python -m citypulse_ai.sqlpilot.mcp \
  --database outputs/historical/samples/2024-01-n1000/citypulse.duckdb
```

This is a newline-delimited JSON-RPC **stdio subset of MCP 2025-03-26**, not a
new network service or full modern MCP SDK. It supports `initialize`, the required
`notifications/initialized` handshake, `ping`, `tools/list` and `tools/call`.
The two tools are `sqlpilot_schema` and `sqlpilot_query` (SQL string only).
Read-only/idempotent annotations do not replace server validation. Execution
errors use `isError`; bad tools/arguments produce JSON-RPC errors. Frames are
bounded at 8 KiB, sessions at 128 messages, requests execute serially, and EOF
ends the server. Clients requiring newer stateless-only MCP need an adapter or
compatible legacy negotiation; prompts/resources/sampling are not implemented.
No MCP client integration certification or real LLM tool call is claimed.

Reference: [MCP tools specification](https://modelcontextprotocol.io/specification/2025-03-26/server/tools).

## Measured evaluation

```sh
PYTHONPATH=src .venv/bin/python -m citypulse_ai.sqlpilot \
  --database outputs/historical/samples/2024-01-n1000/citypulse.duckdb --evaluate
PYTHONPATH=src .venv/bin/python -m pytest tests/test_sqlpilot.py -q
```

The five candidate SQL statements are **hand-authored**, compared with independent
fixed reference queries (for example, summed station-hour starts versus accepted
trip count). Real sample results: **5/5 execution successes, 5/5 correct results,
6/6 unsafe probes rejected**. Median end-to-end process/query latency was **0.607 s**,
maximum **1.047 s** in that run; these are local single-run measurements, not SLOs.
One separately measured sample-query worker peaked at **129,007,616 bytes RSS**
(about 123 MiB), with zero per-process swaps. Host swap was already elevated and
must not be attributed to SQLPilot without evidence.

Tests cover fixed reference results, joins, filters, grouping/top-k, local weekdays
and DST fall-back, hidden-column attacks, write/external/extension/system-schema
attempts, read-only enforcement, row caps, parser version mismatch, macro shadowing,
timeouts, MCP lifecycle and a real stdio subprocess. An inconsistent synthetic
sample deliberately fails independent result checks, proving the evaluation does
not simply equate execution with correctness. A first constant-AST handling bug
was found and fixed before the passing runs; unsupported startup TimeZone config
was moved to trusted post-connect UTC initialization. No failing result was hidden
by skipping tests.

Generated evaluation JSON and databases remain under ignored `outputs/` or test
temporary directories. `.github/workflows/sqlpilot-ci.yml` runs synthetic core tests
on a standard Ubuntu runner using only DuckDB, pytz and pytest, with no credentials,
LLM weights or forecasting dependencies.

## What remains for SQLPilot v1

1. Choose and approve an accessible real backend. The cached local Gemma 3 option
   requires a reachable Ollama service **and safe RAM headroom**; do not start it
   automatically under the present memory pressure. Alternatively use an approved
   existing MCP client/model, or a separately approved hosted endpoint with explicit
   cost/privacy limits. No paid API credentials are assumed.
2. Implement the chosen provider adapter with bounded generation, strict decision
   parsing and clarification behavior; exercise actual question → SQL → rows.
3. Evaluate held-out natural-language questions, ambiguity, unanswerable requests and
   prompt injection; report execution and semantic accuracy separately.
4. Add LangGraph only if bounded repair/tool orchestration requires it. Preserve the
   roadmap for MCP expansion, LangGraph, vLLM and optional QLoRA, but establish a
   measured prompt baseline first. Fine-tuning needs a curated training/evaluation
   set, hardware/budget approval and evidence of an accuracy gap; none is started.
