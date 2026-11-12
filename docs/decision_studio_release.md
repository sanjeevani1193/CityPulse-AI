# Decision Studio release candidate and retirement gates

Local review date: **November 12, 2026**. All five workspaces have user-approved
visual design. Public deployment, the PR/merge and GCP retirement are **pending**.
This document records preparation, not a claim that those actions occurred.

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
three new check functions were exercised successfully; a completed pytest run of
that new file is still for Linux CI to verify.

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

**Clean Linux execution remains unverified until that CI job runs.** Cross-target
wheel resolution and a source-only Mac run are useful checks, not substitutes
for native Linux installation/execution. The Linux job explicitly deselects four
private-source exporter comparisons: three in `test_decision_studio.py` and one
in `test_decision_studio_sql.py`. Those passed locally. All bundled-artifact,
public behavior, statistical and SQL safety checks remain required. With the
current test collection, Linux should run 205 tests and deselect four; these are
expected counts, not a reported CI result.

## Safe GitHub integration: approval required

Read-only GitHub API inspection confirmed `cloudrun-cd.yml` is **active** and
remote `main` is `ff9762570592526a5c2f621ad3cb7546d9126c39`. No push, workflow
dispatch, settings change or GCP operation was performed during this review.

The deployment workflow has a `push` trigger for relevant `main` paths and a
manual dispatch trigger. It can authenticate with WIF, read private inputs and
launch Cloud Build/Cloud Run after CI succeeds. The Decision Studio diff does not
currently match those cloud deployment path filters, but path filters alone are
not the retirement control: future merges or edits could match them.

Before **any** feature push/merge:

1. In GitHub **Actions**, select **XGBoost Cloud Run CD**, open its workflow menu,
   and choose **Disable workflow**. Preserve its YAML, existing runs and IAM.
2. Confirm disabled status to the agent. This changes repository settings and
   must not be done without the requested confirmation/authorization.
3. After confirmation and push approval, push only the feature branch normally:
   `git push -u origin feature/citypulse-decision-studio`.
4. Prepare a PR against `main`; inspect the complete diff and required checks.
   Expected ordinary CI: Inference CI, SQLPilot deterministic core,
   ExperimentLab statistics, and the new Decision Studio offline QA.
5. Do not dispatch **Frozen station100 release verification**: that manual-only
   workflow authenticates to GCS and is unnecessary here. Do not dispatch CD.
6. Review Linux results, any declared exclusions and secrets/file checks. Require
   a separate user approval before merging. Do not force-push or change protection
   rules to bypass failed checks.

Disabling the CD workflow neither shuts down Cloud Run nor disables ordinary CI.
Preserving Git history retains source and engineering evidence, not the private
datasets, fitted weights, MLflow databases or unknown VM disk contents.

## Streamlit Community Cloud configuration

| Setting | Exact value |
|---|---|
| Repository | `sanjeevani1193/CityPulse-AI` |
| Branch | `main`, after the approved PR is merged |
| Entrypoint | `apps/decision_studio/streamlit_app.py` |
| Python | **3.14**, explicitly selected under Advanced settings |
| Dependencies | `apps/decision_studio/requirements.txt` |
| Configuration | Repository-root `.streamlit/config.toml`; no fixed server address |
| Secrets | None; leave the app secrets empty |
| Visibility | Public/anonymous, after deployment approval |
| Suggested subdomain | `citypulse-ai-decision-studio` if available; not reserved or deployed |

Community Cloud executes from the repository root and searches the entrypoint's
directory for dependencies before the repository root. The app's requirements
take precedence over the project-wide `pyproject.toml`, avoiding ML/cloud extras.
No `packages.txt`, model-serving environment or cloud credentials are required.
The entrypoint resolves `src/` from its own path; bundle/report paths also resolve
from module locations, not private workstation paths.

After merge and separate deployment approval, use **Create app** in
`share.streamlit.io`, enter these coordinates, select Python 3.14, and deploy.
Do not use a default Python version without checking it. Monitor build logs,
then test the actual URL in an anonymous session: all pages, selectors, SQL run,
report/CSV downloads, narrow viewport, keyboard focus and sidebar collapse.
Only then record the real URL and public-verification evidence in README.
No public URL or currently active cloud service is certified by local tests.

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

**Retirement is not yet safe to authorize:** independent cloud-only backups,
disk disposition, CD disablement and public-app verification are outstanding.
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

## First remote approval checkpoint

Confirm that **XGBoost Cloud Run CD is disabled at repository level**, and then
authorize pushing this feature branch and opening a PR. No merge is authorized
yet. Linux QA must succeed on the exact PR revision. Public deployment and all
backup/transfers/billing/resource changes remain separate later approvals.
