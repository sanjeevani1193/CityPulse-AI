# SQLPilot: Groq preparation, pending explicit live-call approval

Reviewed October 28, 2026. **No authenticated Groq request or model inference was
performed.** Public documentation establishes an advertised offering, not this
user's account access, remaining quota, current billing state or model eligibility.
Forecasting v1, LangGraph routing, MCP execution and SQL safety controls are unchanged.

## Recommendation and verified public offering

Start with **GroqCloud `openai/gpt-oss-20b`**, held fixed for the nine-case baseline.
This is a hosted model: no model download, GPU or additional local model RAM.
Groq lists it as a production text model with JSON Object/Schema modes and tool use.
It is a practical compact reasoning/coding candidate, **not a measured SQLPilot
accuracy winner**. The larger `openai/gpt-oss-120b` is another published candidate,
not an automatic fallback or a reason to expand this initial evaluation.
[Models](https://console.groq.com/docs/models),
[20B model card](https://console.groq.com/docs/model/openai/gpt-oss-20b).

The published Free Plan table currently lists both GPT-OSS models at **30 RPM,
1,000 RPD, 8,000 TPM and 200,000 TPD**. Limits are organization-wide; exact account
limits/remaining allowance may differ. Verify them in the signed-in Console's
Limits page. Nine questions allow at most eighteen generation attempts, plus prompt
input tokens. This is not a token-count guarantee or a reservation of shared quota.
[Rate limits and 429 behavior](https://console.groq.com/docs/rate-limits).

Groq distinguishes Free and paid Developer tiers. Its current billing documentation
says a valid payment method is required to **upgrade**; it does not establish the
specific onboarding requirements for this account. Stay on Free. Do not add payment
details, upgrade, or assume promotional credit makes a paid account acceptable.
If the Console requires payment/upgrade to obtain access, stop and report it.
[Billing FAQ](https://console.groq.com/docs/billing-faqs).

## What must be confirmed by the user

Before any model call, confirm in your own signed-in Console:

1. The organization is on **Free**, with no paid activation required.
2. `openai/gpt-oss-20b` is available to that organization/API key and the intended
   run fits the current request/token allowance. Creating a key alone proves
   neither inference access nor free billing status.
3. You approve Groq receiving the exact schema and question scope below.
4. You reviewed Data Controls and the current retention terms. Prefer enabling
   Zero Data Retention before approving sample transmission.

The application flags below record these operator attestations; they are **not**
a billing-status API, a hard monetary cap or a way to prove free eligibility.
No configuration flag has been activated during preparation. Groq Free is separate
from GCP trial credits and existing coding-assistant subscriptions; neither proves
Groq API eligibility. Only a Groq key is needed for this endpoint.

## Privacy and outbound data

The model request contains:

- Fixed SQL-generation instructions and the chosen model/settings.
- Your analytical question; the nine-case corpus includes an intentional malicious
  instruction used to test rejection/clarification.
- Only the permitted schema: table names, column names/types and **sample row counts**.
- For one correction, previous generated SQL and a bounded generic execution-error
  category appended to the question. No detailed database exception is sent.

The current schema is `hourly_rides` (start_station_id, hour_start, ride_count;
1,931 rows) and `validated_trips` (started_at, ended_at, start_station_id,
end_station_id, duration_seconds; 1,998 rows). Station IDs remain strings and
NYC calendar calculations require explicit America/New_York conversion.

Its canonical SHA-256, verified through the real MCP schema tool, is:

```text
cbf144235c6a0ee5aedc0d0ce0e80abc0719d5dd5de7c445b1191275ebb710ae
```

The frozen questions are in `tests/fixtures/sqlpilot_questions.json`. The reference
SQL and expected rows are **not** placed in the generation prompt. Query result
rows, raw ride records, the DuckDB file, source filenames, model weights and
forecasting results remain local. Explanations are grounded locally, not sent for
another LLM call. The API key travels only as an HTTPS Authorization credential,
not in the model prompt, traces, UI or Git. Usage/token metadata will still reach
the service; withholding result rows does not make this an offline workflow.

Groq says inference customer data is not retained by default, but reliability/abuse
investigations can log inputs/outputs for up to 30 days (longer if legally required).
Usage metadata is always collected. All customers may enable ZDR in Data Controls;
retained customer data is stored in US GCP buckets. Do not assume EU residency or
unconditional zero retention. Review the actual account setting and current terms;
this preparation does not make an additional model-training-use assurance.
[Groq data policy](https://console.groq.com/docs/your-data).

## Obtain and configure a key securely

1. Open [Groq Console API Keys](https://console.groq.com/keys) yourself and sign in.
2. Verify Free status and limits as above. Create a dedicated SQLPilot key, using
   project/model restrictions if available. Do not test it in Playground yet.
3. Store the key locally in an environment variable or your password manager.
   Never paste it into this chat, a committed file, screenshot or terminal command
   containing its literal value.

For the project's zsh shell, this reads it without terminal echo or shell history:

```sh
read -rs 'GROQ_API_KEY?Groq API key (hidden): '
printf '\n'
export GROQ_API_KEY
export SQLPILOT_PROVIDER=groq
export SQLPILOT_MODEL=openai/gpt-oss-20b
```

These settings **alone do not authorize or enable generation**. Leave approval
flags unset until explicit user approval. After approval and account verification,
set these in the same terminal (not in a tracked configuration file):

```sh
export SQLPILOT_PROVIDER_APPROVED=groq
export SQLPILOT_FREE_ALLOWANCE_VERIFIED=yes
export SQLPILOT_APPROVED_SCHEMA_SHA256=cbf144235c6a0ee5aedc0d0ce0e80abc0719d5dd5de7c445b1191275ebb710ae
export SQLPILOT_QUESTIONS_APPROVED=yes
export PYTHONPATH=src
export SQLPILOT_MCP_PYTHON="$PWD/.venv/bin/python"
```

The default Groq endpoint is exactly
`https://api.groq.com/openai/v1/chat/completions`; no custom endpoint approval is
needed for that fixed profile. A different provider/endpoint requires its own
explicit approval and the existing `openai-compatible` configuration gates.
If the sample schema/counts change, its hash changes and generation fails closed.
The questions flag attests approval of the actual question scope, not permission
to silently transmit unrelated or sensitive questions later.

## Adapter readiness and bounded request profile

The existing `Backend.generate(question, schema)` adapter and graph/MCP execution
path are reused. Groq GPT-OSS requests now use:

- JSON Object mode; generated decisions remain independently validated.
- `temperature=0`, `reasoning_effort=low`, `include_reasoning=false`.
- `max_completion_tokens=1024`, a fixed bound shared by reasoning and final output;
  truncation is treated as failure, not a completed SQL decision.
- Existing HTTPS endpoint restrictions, no redirects/proxy inheritance, 12-second
  HTTP timeout, response-size cap, two graph generation attempts, and unchanged
  SQL checks/resource limits.

Groq documents that GPT-OSS **does not support `reasoning_format`**; that parameter
is not sent. Its completion parameter replaces the deprecated `max_tokens` for
this profile. Other approved compatible providers keep their existing generic
request profile. No tool/browser/code-execution capability is requested from Groq;
MCP tool execution remains local under LangGraph's control.
[Reasoning parameters](https://console.groq.com/docs/reasoning),
[API reference](https://console.groq.com/docs/api-reference).

Compatibility is verified with mocked HTTP responses only. Actual free-account
access, model availability, quota and live response quality remain unverified.
A length-limited/abnormal finish, bad JSON or provider error fails safely.

## Prepared evaluation, still NOT executed live

There are six answerable cases and three clarification cases (ambiguity,
unobserved physical demand, malicious instructions). The evaluator reports:

- Answerable execution successes / six eligible answerable cases.
- Correct result rows / six, against reference SQL executed through MCP.
- Clarification statuses / three, with separate unsafe-question decline count.
- Query-tool errors (including binding errors), **not mislabeled as unsafe-SQL
  rejection accuracy**. Core attack rejection remains a separate deterministic test.
- Per-case and median/max latency, attempts, model/provider identity, question-spec
  hash, planned/evaluated/not-run counts, and whether real generation occurred.

Clarification content needs human review in addition to the automated status check.
A valid executed SQL query is not automatically a semantically correct answer.
Partial reports distinguish evaluated category counts from planned counts and
never treat unevaluated questions as successful.

Live batches wait at least **65 seconds between questions**, outside the graph's
per-question timeout. This conservative pacing does not guarantee quota availability,
especially with shared traffic or correction prompts. The batch stops on provider
HTTP errors, including 401/403/429, and records unevaluated cases. It never upgrades
billing or automatically retries an HTTP failure. SQL correction remains limited
to one retry through the original safety core.

Only after approval, first run one benign question in the UI to verify actual
connectivity/identity/latency. Then use the unchanged nine-case corpus:

```sh
.venv-sqlpilot/bin/python -m citypulse_ai.sqlpilot.evaluate_graph \
  --database outputs/historical/samples/2024-01-n1000/citypulse.duckdb \
  --output outputs/sqlpilot/groq-20b-live-evaluation.json
```

Expect roughly nine minutes or more for a paced nine-question batch, not an
instant test. Reports are create-only and gitignored. Until approved, this command
returns `live_backend_pending` with zero model calls/cases. No live accuracy score
or Docker readiness is claimed by this preparation milestone.
