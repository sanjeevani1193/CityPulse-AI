# XGBoost Cloud Run CD — manual deployment verified

The approved bootstrap is complete: **14 additions total, zero changes or
deletions**. The first apply created 11 additions before GCP rejected a pool
display name longer than 32 characters. After shortening only the display name
to `CityPulse CD WIF`, a refreshed plan applied exactly the three remaining
additions. No completed resource was recreated. A regression test now checks
this display-name limit. Private original, partial and resumed Terraform state
backups are retained locally; never reuse the stale pre-apply plan.

Both approved private validation JSON files were uploaded create-only and their
generation-pinned bytes were verified. The manifest records request generation
`1793459483219844` and reference generation `1793459483478652`.
The one manual GitHub Actions deployment passed. Automatic push deployment
remains disabled in code. [Initial deployment and cleanup](cloudrun_xgboost.md).

## Verified manual run — October 31, 2026

- Deployment source: `61bb8dd6d963c74f6f3e672dd34bd833ab7e2260`.
- [Successful manual CD run](https://github.com/sanjeevani1193/CityPulse-AI/actions/runs/41213999783).
- All three required CI suites passed on that exact commit. The push-triggered
  CD run was skipped; the single manual run passed its gate, WIF authentication,
  five-input recovery, build, rollout and HTTP verification.
- Cloud Build: `7f9caa9e-7fae-40b4-9fc9-cad8b00fe9dd`, **SUCCESS**, 107.39 seconds;
  exactly one new build, no retry.
- Immutable digest: `sha256:23b97afb9b6d5b6d3577cb2d974e459af8a245a1677eb3ec6d9f664a8ecf0cd8`.
- Registry image size: **218,131,742 bytes** (~208.03 MiB); two retained versions,
  including the original rollback image. No image deletion occurred.
- Ready revision: `citypulse-xgboost-00002-ss2`.
- Authenticated URL: https://citypulse-xgboost-lw3wuexzqq-ey.a.run.app.

| Real runner check | Result |
| --- | --- |
| `/health`, `/stations`, `/models` | 200; model ready, exact 100 station IDs, XGBoost only |
| `/predict` | 200; 24 forecasts, correct UTC/NYC-local timestamps and model identity |
| Saved validation reference | **0.0 maximum absolute prediction error** |
| Repeated request | Identical response |
| Future observation | 422 |
| Anonymous health | 403 |

The runner's complete bounded verification took **2.334 seconds**, not a cold
start or load benchmark. Cloud Run logged prediction request latencies of
0.355/0.242 seconds. Reviewed 21 revision log entries contained zero
ERROR-or-higher entries. The post-CD Terraform plan reports **no changes**.
Only image changed in the service spec; generated operation/last-modifier audit
metadata changed as expected. CPU, memory, identity, scaling, concurrency and
startup settings match the initial deployment. Original verification WIF and
stopped historical VM snapshots remain identical. No forecasting/data/ETL changes.

Sanitized `deployment.json` and `verification.json` are retained in the linked
Actions run for 14 days; private evidence and state backups remain under ignored
`outputs/cloudrun-xgboost/phase4/`. State, model bytes, inputs, credentials and
source archives were not published as GitHub artifacts.

Build list-cost estimate: **$0.00537** at $0.003/minute, before allowances/credits
and ancillary charges. This is not an actual invoice or updated trial balance.
The run made six authenticated requests plus one anonymous denial. Registry,
source storage and logging continue to have retention costs; none were pruned.

Manual CD is verified. **Automatic CD is still disabled**. Enabling it requires
separate approval, a reviewed main-push job condition and one approved relevant
push to verify that path. Do not dispatch another build merely for these docs.

## Small deployment path

Relevant `main` push (automatic mode initially off), or manual workflow → wait
for all three existing CI suites on the **exact current main SHA** → GitHub OIDC
→ separate WIF provider → dedicated deployer → restore five pinned small inputs
→ existing create-only packager → private source archive → one amd64 Cloud Build
→ immutable registry digest → image-only PATCH of the existing Cloud Run service
→ authenticated HTTP and saved-validation parity checks.

No Terraform apply happens in Actions. Infrastructure stays under local-state
Terraform. The existing builder and role-free runtime identities are reused.
The six-model application, forecasting artifacts, original verification WIF and
all historical resources are untouched.

### Trigger and CI rules

`.github/workflows/cloudrun-cd.yml` supports `workflow_dispatch` from main.
The initial job condition permits **workflow_dispatch only**; pushes cannot
reach the cloud job, regardless of repository variables. Enabling automatic
deployment requires a separately reviewed workflow-condition change after the
manual run passes. Push path filters cover the packaged Python directories, XGBoost Docker/
build configuration, frozen input manifest, CD script and workflow; docs-only
changes do not build. Because the existing packager includes shared forecasting
modules, changes to those packaged modules also count as relevant.

The gate waits at most 15 minutes for successful **Inference CI**, **SQLPilot
deterministic core** and **ExperimentLab statistics** push runs on that SHA.
Failures, cancellation, missing runs or non-main code stop deployment. A newer
main SHA makes an older queued/building deployment fail closed. One concurrency
group serializes manual and automatic runs without cancelling active builds.
Only the deploy job gets `id-token: write`; checkout credentials are not persisted.
All actions are pinned to reviewed commit SHAs. CI continues to run deterministic
CD tests without cloud credentials, model artifacts, inference or billable work.

## Frozen private inputs

`infra/cloudrun-cd-inputs.json` pins the existing three model/preprocessing
objects by generation, byte size and SHA-256 (2,531,795 bytes total). Live metadata
checks confirmed all three pinned generations and lengths still exist. Runner
restoration checks downloaded SHA-256 bytes, then the existing packager checks
model hashes and the 100-station vocabulary again. No retraining, original-data
restore, benchmark rerun or other model download is needed.

Two existing **validation-only** demo files received a one-time approved private
upload, without changing their bytes:

| Local source | Proposed existing-bucket object | Bytes |
| --- | --- | ---: |
| `outputs/forecasting-v1/demo-request.json` | `cloudrun-cd/validation/demo-request.json` | 28,209 |
| `outputs/forecasting-v1/demo-references.json` | `cloudrun-cd/validation/demo-references.json` | 4,376 |

Their SHA-256 values are recorded in the manifest. Upload with a create-only
`ifGenerationMatch=0` precondition, verify remote metadata and bytes, then fill
in their returned generations **before committing the workflow**. Both returned generations are now pinned. Any missing/invalid generation
causes restore to fail before network I/O/build. Never substitute latest/unpinned objects. The demo contains one
approved station's 336 validation-history hours and existing saved reference
forecasts; it is private data, not a GitHub fixture or public Actions artifact.
Only XGBoost reference values are used. No held-out targets are fetched/scored.

The runner creates a unique private source object under `cloudrun-build-source/cd/`
with a create-only precondition. The existing builder can read that prefix.
Credentials are outside the allowlisted context. Artifacts, source archives and
request histories are never uploaded to GitHub artifacts; only sanitized
`deployment.json` and `verification.json` receipts are retained for 14 days.
Ephemeral runner storage is discarded by GitHub. No automatic cloud cleanup is
introduced; old image/source retention needs separate reviewed approval.

## Reviewed IAM and Terraform ownership

Terraform plan: **14 additions, 0 changes, 0 deletions**. No API enablement is
required: IAM Credentials, IAM, STS, Cloud Build, Registry and Run are enabled.
The existing WIF provider is not imported, edited or broadened.

| New access | Scope and reason |
| --- | --- |
| Separate pool/provider `citypulse-cloudrun-cd/github-main` | Repo ID `1410625891`, owner ID `91110975`, main ref, exact CD workflow path, push/manual events only |
| `citypulse-xgb-deployer` service account | Short-lived impersonation only, no keys |
| `roles/iam.workloadIdentityUser` | Only that pool's immutable repository-ID principal set; enables short-lived access/ID tokens |
| Custom build role | Project: `cloudbuild.builds.create`, `cloudbuild.builds.get`, `serviceusage.services.use` |
| `roles/iam.serviceAccountUser` | Only the existing builder and runtime service accounts |
| Custom service role | Only `citypulse-xgboost`: `run.services.get`, `run.services.update` |
| `roles/run.invoker` | Only the existing authenticated service, for HTTP checks |
| `roles/artifactregistry.reader` | Only the existing private repository; no write/delete grant to deployer |
| `roles/storage.objectViewer` | Condition on exactly the five frozen input object names; no bucket-wide listing |
| `roles/storage.objectCreator` | Condition on `cloudrun-build-source/cd/` only; cannot overwrite/delete/read historical objects |

The 14 additions comprise pool, provider, service account, two custom roles,
project build binding, federation binding, two actAs bindings, two service
bindings, registry binding and two conditional bucket bindings.

IAM cannot constrain `run.services.update` to a single JSON field. The workflow
therefore PATCHes only `template.containers`, preserving the complete existing
container settings except image, with the service's etag for concurrency control.
It has no create/delete-service or set-IAM-policy permission. Trusted-main review
is essential: build-create and service-update permissions are meaningful authority,
not a hard monetary or image-field security boundary. Build config explicitly
selects the existing limited builder and the workflow performs one submit only.

Terraform ignores **only `template[0].containers[0].image`**, so later CD image
updates are not reverted. All other settings, IAM and resource drift remain
managed. Preserve the current ignored `deployment.auto.tfvars.json` and state;
`deploy_service` must remain true. No state/plan/credentials go to GitHub.

## Local validation and deployment procedure

```sh
terraform -chdir=infra/terraform fmt -check
terraform -chdir=infra/terraform validate
terraform -chdir=infra/terraform plan -input=false -out=phase4-review.tfplan
PYTHONPATH=src .venv/bin/python -m pytest tests/test_cloudrun_cd.py tests/test_cloudrun_xgboost.py tests/ci/test_station_contracts.py -q
.venv/bin/python scripts/check_ci.py
git diff --check
```

Measured local verification: **67 passed, 0 failed/skipped, 1 training test
explicitly deselected**, covering CD, Cloud Run contracts, station contracts and
causal XGBoost features. The XGBoost import test required the existing documented
macOS per-process setting (no environment installation/change):

```sh
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.14/site-packages/sklearn/.dylibs" PYTHONPATH=src .venv/bin/python -m pytest tests/test_cloudrun_cd.py tests/test_cloudrun_xgboost.py tests/ci/test_station_contracts.py tests/test_station_xgboost.py -k 'not native_tiny_training' -q
```

Terraform fmt/validate and the refreshed plan passed. Static checks covered 159
Python files; whitespace and relative documentation links passed. Existing
Starlette/httpx deprecation warning remains. `actionlint` is not installed; YAML
and deterministic workflow-policy tests passed, but GitHub runner, WIF, build
and real cloud CD parity subsequently passed in the manual run recorded above. Original Terraform state and
unrelated README draft checksums are unchanged. Private reviewed plan text/JSON
is under ignored `outputs/cloudrun-xgboost/phase4/` with owner-only permissions.

Initial approval and the manual sequence below have been completed. Future
updates still require reviewed plans; do not rerun the initial bootstrap or the
stale saved plans. A saved plan by itself is **not authorization to apply**.
Before proceeding, approve: the 14 additive IAM/WIF resources; the two private
demo uploads; committing/pushing the reviewed files; and **one** manual CD build,
image rollout and bounded HTTP verification. Refresh the plan if inputs change,
back up local state privately, then apply only the approved plan. Capture the
uploaded demo generations, update the manifest and repeat local checks. Push
normally with the manual-only job condition intact, wait for CI, then manually dispatch:

```sh
gh workflow run cloudrun-cd.yml --ref main
```

Use GitHub's Actions UI if `gh` is not installed. Check sanitized receipts,
build logs, image digest, authenticated endpoints, exact station IDs, 24 elapsed
UTC hours, reference parity (rtol 1e-5/atol 1e-4), repeated output, future rejection
and anonymous denial. A failed build or parity check leaves the job failed;
there is no automatic rebuild or rollback. Inspect and request approval before
another billable attempt. Finally run a fresh local Terraform plan to confirm
no unintended infrastructure drift.

After successful verification, keep the manual-only condition until a separate
automatic-CD approval. Enabling push deployment requires a reviewed condition
change and one subsequent relevant main-branch change to verify it; neither is
part of this manual run.

## Costs and limits

No new always-on compute or storage service is created. Each relevant approved
run adds one E2_MEDIUM build (900-second Cloud Build timeout), a source archive
(~3 MB previously), a new image version (~208 MiB previously), logs, and a few
Cloud Run requests. At $0.003/build-minute, the build ceiling is approximately
**$0.045 per job** before allowances, startup/ancillary costs and credits; the
prior build took 75 seconds (~$0.0038 list price). This is not a hard spending
cap. CI/runner use depends on GitHub account allowances. Registry/source storage
accumulates until separately approved cleanup. No repeated retries are configured.
Cloud Run stays authenticated, request-billed, min 0/max 1, 0.5 CPU/512 MiB GEN1.
Free Trial credit, scaling limits and budget alerts are not hard cost guarantees.

References: [Google WIF deployment-pipeline restrictions](https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines),
[Cloud Build permissions](https://docs.cloud.google.com/build/docs/iam-roles-permissions),
[Cloud Run PATCH API](https://docs.cloud.google.com/run/docs/reference/rest/v2/projects.locations.services/patch),
[Cloud Build pricing](https://cloud.google.com/build/pricing).
