# Decision Studio public release and retirement gates

Release documentation updated **November 12, 2026**. The project owner reports
successful incognito-browser verification of all five publicly deployed workspaces:
Overview, Forecast Explorer, Model Comparison, SQLPilot, and ExperimentLab & Methodology.

**[Live Decision Studio](https://citypulse-ai-decision-studio.streamlit.app/)**

[PR #2](https://github.com/sanjeevani1193/CityPulse-AI/pull/2) was merged normally
as `4eda1751ccdc72d0778995fc1f9ed480a91cd4cf`, preserving the five development
commits and feature branch. Public access is owner-reported manual evidence;
automated native Linux evidence is recorded separately below. No new browser
automation, model inference or cloud operation was performed for this documentation
update. GCP backup/retirement remains **pending**, not authorized by public release.

## Public runtime and post-merge QA

The public Streamlit app presents verified historical forecasts, saved six-method
benchmark results and diagnostics, fixed read-only SQL over a small stripped sample,
and saved fully synthetic experiment outcomes. It does not load forecasting models,
generate SQL with an LLM, run MCP/agent sessions, simulate new experiments or call GCP.
Interactivity comes from selectors/charts and actual curated local SQL execution.
The historical authenticated XGBoost Cloud Run deployment is a separate system;
its disabled CD workflow does not imply its resources or billing have been retired.

| Post-merge workflow | Verified result |
|---|---|
| [Decision Studio offline QA](https://github.com/sanjeevani1193/CityPulse-AI/actions/runs/43211528431) | Ubuntu 24.04/Python 3.14; clean install and `pip check`; **205 passed, 4 explicitly deselected, zero failures/skips**; real HTTP/WebSocket checks of all five pages, curated SQL and checksum-identical HTML download |
| [Inference CI](https://github.com/sanjeevani1193/CityPulse-AI/actions/runs/43211528389) | Configuration checks, **130 gateway + 17 worker tests** passed; two nonblocking test-client deprecation warnings |
| [SQLPilot deterministic core](https://github.com/sanjeevani1193/CityPulse-AI/actions/runs/43211528423) | Core, application and container checks passed |
| [ExperimentLab statistics](https://github.com/sanjeevani1193/CityPulse-AI/actions/runs/43211528496) | **19 tests** and generated-report checksum verification passed |

All eight post-merge jobs succeeded on `4eda175`. The four explicit exclusions
compare exports with private retained research sources; those checks passed locally.
CI uses public-safe bundled artifacts and never restores those private inputs.
The native HTTP smoke measured about 229 MiB peak server RSS, took 2.72 seconds,
and stopped its owned server. These are engineering smoke tests, not a public
load test, uptime guarantee, new forecast benchmark or live LLM evaluation.

The owner's anonymous browser checks establish public access to the five pages.
They are not represented as a newly automated public-URL test, continuous monitoring
or a pixel-level accessibility certification. No screenshot evidence was generated
during this documentation update.

## Local release evidence

Milestone 4: `96d380d3354902f59cc93ef10937a00d51d7c440`.
Milestone 5: `15afb7f9b1722d926210debdcc2192d203890521`
(`feat: add ExperimentLab and methodology workspace`). The latter committed 15
reviewed files; its working tree was clean immediately after the commit.

| Verification | Actual outcome |
|---|---|
| Overview/comparison and Forecast Explorer suites | 74 passed |
| Curated SQL workspace suite | 39 passed |
| Experiment workspace suite | 32 passed |
| Original SQLPilot safety regressions | 42 passed |
| Entire original ExperimentLab suite | 19 passed, including all 16 previously excluded cases |
| Combined completed groups | **206 passed; no failures or skips in those completed groups** |
| Three new release checks | Hosting settings, dependency discovery and source-only five-page smoke assertions passed; see resource caveat below |
| Real local Streamlit HTTP/WebSocket smoke | Root/health 200; five pages rendered, 24 forecast points and TFT band checked; curated SQL returned 1,998 starts; downloaded HTML matched the saved report SHA-256 |
| App dependencies | CPython 3.14 Linux x86_64 binary-wheel resolution succeeded for 36 distributions; nothing installed by that dry run |

The initial single-process combined test attempt was **interrupted**, not passed:
host-wide swap use grew more than 256 MiB. An added pytest-based source-only check
was interrupted by the same guard. Neither established an application failure,
and swap activity cannot be attributed to these processes alone. The original
tests completed in separate bounded groups. The new source-only assertion then
passed with a standard-library parent instead of a second pytest runtime. All
three new check functions were exercised successfully; the subsequent native Linux
pytest run also passed, as recorded in the post-merge evidence above.

Safeguards were retained: 512 MiB owned RSS, 256 MiB host swap-use growth, no
critical memory pressure, 15 GiB free-disk reserve, ten-minute verification bound.
The successful HTTP smoke took 3.39 seconds, with about 178 MiB server peak RSS;
the sampled combined verifier/server peak was 357 MiB. Its host swap-use growth
was about 128 MiB and host pressure stayed warning-level. All owned processes
exited. These samples do not prove memory safety under public concurrent load.

The source-only test uses an isolated interpreter, copies only the runtime code,
packaged evidence and report, and denies outbound sockets. It renders all five
pages and actually executes curated SQL without private `outputs/`, credentials,
an editable project install, model runtimes or GCP. Child SQL execution retains
the existing AST allowlist, read-only DuckDB, external-access prohibition, memory
and timeout limits. Public users cannot submit arbitrary SQL in this Studio.

The 16 original cases are affordable because they simulate only 100 synthetic
station units per replicate. They ran in 2.41 seconds at approximately 93 MiB
sampled RSS, using temporary pytest outputs. The frozen research JSON/report
were not regenerated or overwritten.

### Necessary QA corrections

- Removed global `server.address = "127.0.0.1"` from `.streamlit/config.toml`,
  allowing the hosting platform to select its interface. Local launch commands
  and the smoke verifier still explicitly bind loopback. CORS/XSRF protections
  remain enabled by default; the visual palette is unchanged.
- Extended actual HTTP verification to download the original HTML and compare
  its bytes by SHA-256, rather than merely checking for a download button.
- Added `tests/test_decision_studio_release.py` for hosting configuration,
  dependency discovery and a source-only, network-denied execution check.
- Prepared `.github/workflows/decision-studio-ci.yml` for clean Ubuntu/Python
  3.14 testing after the authorized PR push. This non-cloud workflow has only
  `contents: read`, an eight-minute timeout, no secrets/OIDC/GCP steps, and no
  deployment job. Existing cloud workflows are unchanged.

**Clean Linux execution is now verified by the post-merge CI job above.** The
earlier cross-target wheel resolution and source-only Mac run were preliminary
checks, not substitutes for native execution. The Linux job explicitly deselects four
private-source exporter comparisons: three in `test_decision_studio.py` and one
in `test_decision_studio_sql.py`. Those passed locally. All bundled-artifact,
public behavior, statistical and SQL safety checks remain required. With the
current test collection, the measured Linux result is **205 passed, 4 deselected**.

## Completed GitHub integration and deployment safety

The initial Phase B inspection found `cloudrun-cd.yml` active before integration.
The owner subsequently disabled it, and GitHub API checks before push, before
merge, after merge and during this documentation review confirmed
**`disabled_manually`**. Main now contains `4eda175`; no GCP deployment or
authenticated cloud verification workflow was triggered by the integration.

The deployment workflow has a `push` trigger for relevant `main` paths and a
manual dispatch trigger. It can authenticate with WIF, read private inputs and
launch Cloud Build/Cloud Run after CI succeeds. The Decision Studio diff does not
currently match those cloud deployment path filters, but path filters alone are
not the retirement control: future merges or edits could match them.

The approved integration followed this sequence; retain it as historical safety evidence:

1. The owner disabled **XGBoost Cloud Run CD** through GitHub Actions, preserving
   its YAML, runs and IAM; the API state was independently checked.
2. After explicit approval, the reviewed feature branch was pushed normally.
3. PR #2's 45 files, artifact hashes, five commits and non-cloud CI were verified.
4. After separate merge approval, a normal merge preserved the five commits and
   retained the feature branch. Local main was safely fast-forwarded.
5. All applicable post-merge non-cloud CI passed. Neither CD nor the manual-only
   **Frozen station100 release verification** workflow was dispatched.

Keep cloud CD disabled. Do not re-enable it or dispatch cloud-authenticated checks
as part of documentation or Community Cloud updates. Future pushes/merges still
require the user's authorization; do not force-push or bypass failed checks.

Disabling the CD workflow neither shuts down Cloud Run nor disables ordinary CI.
Preserving Git history retains source and engineering evidence, not the private
datasets, fitted weights, MLflow databases or unknown VM disk contents.

## Streamlit Community Cloud configuration

| Setting | Exact value |
|---|---|
| Repository | `sanjeevani1193/CityPulse-AI` |
| Branch | `main` |
| Entrypoint | `apps/decision_studio/streamlit_app.py` |
| Python | **3.14**, explicitly selected under Advanced settings |
| Dependencies | `apps/decision_studio/requirements.txt` |
| Configuration | Repository-root `.streamlit/config.toml`; no fixed server address |
| Secrets | None; leave the app secrets empty |
| Visibility | Public/anonymous, verified by the project owner in an incognito browser |
| Actual URL | `https://citypulse-ai-decision-studio.streamlit.app/` |
| Custom subdomain | `citypulse-ai-decision-studio` |

Community Cloud executes from the repository root and searches the entrypoint's
directory for dependencies before the repository root. The app's requirements
take precedence over the project-wide `pyproject.toml`, avoiding ML/cloud extras.
No `packages.txt`, model-serving environment or cloud credentials are required.
The entrypoint resolves `src/` from its own path; bundle/report paths also resolve
from module locations, not private workstation paths.

The deployment coordinates above match the merged repository. For reproducing
the setup, Community Cloud's **Create app** flow accepts these repository/branch/
entrypoint values; Advanced settings selects Python 3.14 with secrets left empty.
The requirements file is discovered automatically beside the entrypoint.
Do not infer public success from local tests alone: public access was separately
confirmed by the owner. Additional checks of downloads, narrow viewports, keyboard
focus and sidebar behavior remain useful; no new automated public test is claimed
here. This documentation task does not create, redeploy or rename an app.

Official setup: [file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization),
[dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
[Python selection and deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## GCP backup and retirement preparation: not executed

No GCP listing, download, VM operation, billing change, IAM change, snapshot,
Terraform refresh/apply/destroy, build or deployment was performed in this phase.
All cloud identities, sizes and generations below come from **historical local
receipts**, not a fresh inventory. They must be revalidated after backup approval.

| Evidence/artifact | Size from receipts | Preservation status |
|---|---:|---|
| Historical raw objects (127) | 16.717 GiB | Do not assume publisher URLs will reproduce identical bytes; preserve approved original generations |
| Curated historical inputs/snapshots (175) | 6.416 GiB | Essential for exact reconstruction; only part of history exists on this Mac |
| Historical checkpoints/recovery objects (30) | 7.372 GiB | Preserve audit identities, manifests, database/view dependencies and recovery metadata |
| Final station100 dataset recovery ZIP | 110,793,351 bytes | No local ZIP found; generation `1792843627171250` recorded; essential cloud-only dataset backup |
| Final XGBoost recovery ZIP | 3,622,070 bytes | ZIP not found locally; selected serving model/preprocessing copies were verified locally |
| Final six-epoch TFT recovery ZIP | 963,856 bytes | Local archive SHA-256 and all ZIP CRCs verified; 19 members; still needs an independent off-device copy |
| Final MLP recovery ZIP | 4,429,179 bytes | ZIP not found locally; selected serving checkpoint/preprocessing copies verified locally |
| Final LSTM recovery ZIP | 4,576,517 bytes | ZIP not found locally; selected serving checkpoint/shared preprocessing copies verified locally |
| Final prediction/evaluation and MLflow recovery | Not yet exhaustively inventoried | Preserve exact keys/predictions, selections, histories and tracking stores, not just displayed scores |
| Retained 100 GiB VM disk | Actual used bytes unknown | Local receipts do not prove every disk file is recoverable; not cleared for deletion |

The 332 historical objects total **32,755,064,450 bytes (30.506 GiB)**. A broader
candidate list built from seven retained publication/inventory reports covers
**452 distinct object generations, 32,945,088,350 bytes (30.683 GiB)**. This is
not a complete bucket/version inventory and does not establish current existence
or independent backup. The private local candidate plan is ignored at
`outputs/decision-studio-verification/release-20261112/backup-candidates.json`.

Seven serving files matched their generation-pinned provision receipt locally,
including all four selected model weights and their required preprocessing.
That protects inference inputs on this Mac, not complete training optimizer/RNG
state, historical data or MLflow recovery. Docker images/environments can be
rebuilt from source plus verified artifacts; original random fits and provenance
should be preserved instead of relying on retraining to reproduce them.

### Capacity, costs and backup order

Approximately **124 GiB was free locally** during this review. Only the internal
Macintosh volume was visible; no external backup drive was established. The
known 30.683 GiB compressed/object payload fits locally while retaining 15 GiB,
but the final inventory and restoration footprint must be bounded first. Use
sequential transfers and one-at-a-time restore probes; reserve roughly 65 GiB
for the known payload plus an equally sized working allowance. Larger expansion
must be estimated from archive member sizes before extraction. An encrypted
128 GiB-or-larger external destination is preferable, plus an independent second
copy of irreplaceable metadata/models. Do not fill the Mac merely to retire GCP.

At an illustrative **$0.12/GiB** internet-transfer rate, the known candidate
payload is approximately **$3.68**, plus object operations and any applicable
retrieval/other charges. This is an estimate, not measured usage or approval.
Check actual source storage classes, destination, account currency, allowances
and current trial status before transferring. Do not assume credits or budget
alerts are spending caps. [Official storage pricing](https://cloud.google.com/storage/pricing).

Proposed sequence, requiring separate approval:

1. Approve a fresh **read-only** live object/version inventory, a backup destination
   and transfer budget. Reconcile every required receipt/generation, including
   final MLflow supplements and all selected model/recovery dependencies.
2. Preserve station100 recovery, final models/preprocessing, exact saved
   predictions, reports and MLflow state first. Then back up historical
   raw/curated/checkpoint objects needed for the chosen recovery guarantee.
3. Download only a reviewed manifest, generation-pinned and create-only; never
   recursively copy the entire bucket without a capacity/cost review. Example
   **not executed** (destination must be an approved empty staging directory):

   ```sh
   gcloud storage cp \
     'gs://citypulse-ai-511315-data/datasets/tft/accepted_recorded_starts_v1/station100-20261024/dataset-recovery.zip#1792843627171250' \
     /APPROVED-EMPTY-DESTINATION/dataset-recovery.zip.partial
   ```

   Require exact size **110,793,351** and SHA-256
   `5e7642bbb72c5950fa9017770701b14b7289d69bf919560ebe3f5789c7035a49`
   before promoting that new partial file. Use streaming hashes, exact generations,
   and an itemized receipt for every object; do not overwrite existing backups.
4. Test archive CRCs and member manifests, reject unsafe member paths, and restore
   into an empty directory with existing recovery readers. Check dataset identity,
   cohort/window keys, schemas and internal hashes. MLflow recovery needs store
   integrity/run identity checks; model reload probes need the documented CPU
   environment. A successful download/CRC check alone is not sufficient.
5. Resolve the stopped disk's unknown contents. Metadata cannot inspect its
   filesystem. A later temporary, read-only content inspection requires its own
   approved resource plan; alternatively explicitly accept unverified residual
   disk files as expendable after known research backups are independently tested.
   Do not start a VM, create a snapshot or delete the disk in this phase.

### Terraform and independent recovery

The encrypted directory `~/Documents/CityPulse-Backups/2026-11-02-39563c77` remains
outside Git. Its 17,461-byte AES-256-GCM archive still matches the saved SHA-256
receipt. The original receipt records authenticated decryption and byte-verified
restoration; no new decryption/password retrieval was attempted in this phase.
Current Terraform state and `.backup` exist locally and remain ignored. Copy
the encrypted directory to independent encrypted storage, and preserve the
Keychain/password through a separate trusted password-manager or Keychain backup.
Do not put plaintext state or encryption credentials in GitHub or GCS. This
archive does not certify the old VM disk contents or current cloud drift.

### Billing-shutdown gate and manual procedure

**Retirement is not yet safe to authorize:** independent cloud-only backups and
unknown disk contents remain unresolved. CD disablement and owner-reported public
app verification are complete; neither substitutes for research-data backups.
Before recommending shutdown, record completion of each gate, preserve current
state/infrastructure documentation, and review final charges/remaining resources.

Only after separate explicit authorization:

1. Open Google Cloud Console **Billing → My projects**.
2. Locate project **citypulse-ai-511315** and double-check its identity.
3. Open that project's Actions menu and choose **Disable billing**.
4. Review the warning and confirm only for that project; do not close a shared
   billing account or affect another project.
5. Verify the project is unlinked and retain the confirmation and final billing
   report; no new workload/build/restore should depend on that project afterward.

Disabling project billing stops Google Cloud services/billing for that project;
some resources may be removed irrecoverably. It is **not a preservation method**
and does not erase charges incurred earlier, which can appear later. GitHub and
the offline Community Cloud app are separate systems, but their independence must
be verified before shutdown. [Official impact, permissions and manual steps](https://docs.cloud.google.com/billing/docs/how-to/modify-project#disable_billing_for_a_project).

## Remaining approval checkpoints

GitHub integration and public deployment are complete. These documentation changes
are prepared on a separate branch and require approval before any commit/push or
merge. All backup transfers, disk inspection/disposition, resource deletion,
infrastructure changes and billing shutdown remain separate later approvals.
