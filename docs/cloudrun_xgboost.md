# Authenticated XGBoost-only Cloud Run deployment

**Phase 2: implementation and read-only planning only. Not deployed.**
No API was enabled, build submitted, image pushed or Terraform apply executed.
No service URL exists yet. The six-model `api.station_app` and its original
Dockerfiles/Compose remain unchanged. XGBoost is the intentional first deployed
candidate (best recorded six-way held-out MAE); there is no retraining or test
benchmark rerun in this task.

## Exactly three artifacts

Paths below are relative to ignored `outputs/forecasting-v1/artifacts/`:

| File | Bytes | Purpose and existing code |
| --- | ---: | --- |
| `preprocessing.json` | 24,894 | Frozen 100-station vocabulary and shared train-fitted scaler/schema metadata; `ArtifactStore.__init__` → `features.final_station_dataset.load_preprocessing` |
| `xgboost/model.ubj` | 2,502,679 | Trained 300-round booster; `StationRuntime.load('xgboost')` → `xgboost.Booster.load_model` |
| `xgboost/preprocessing.json` | 4,222 | Original 48-feature order and station codes; checked against the booster and common vocabulary in the same loader |

Total: **2,531,795 bytes**. Exact size/SHA-256 identities come from committed
`api/station_artifacts.json`; the verified files already exist locally. **XGBoost
uses raw features, not the shared count scalers.** The shared file remains needed
by the existing context/window preparation; its presence does not refit or apply
MLP normalization. No MLP/TFT/LSTM weights, dataset panels or credentials are
included in the image/source bundle.

## Minimal application and image

`api.cloudrun_xgboost` reuses `StationRequest`, `StationResponse`, `StationRuntime`
and the existing feature builder. Only model `xgboost` is accepted (default);
requests for other models get 422 without importing their runtimes. The original
six-model API continues to expose all six. `/health`, `/stations`, `/predict`
retain their existing successful request/response formats; `/models` lists only
XGBoost here. Startup loads once, fails closed on invalid artifacts and makes
health return 503. No TensorFlow worker is started or contacted.

The new module starts one Uvicorn worker on `0.0.0.0`, reading Cloud Run's `PORT`
(default 8080). Cloud Run terminates TLS and enforces authenticated invocation.
The image uses the existing pinned multi-architecture Python 3.14.7 base digest,
a Linux-amd64 build guard, `xgboost-cpu`, libgomp and pinned API/data dependencies.
No PyTorch, TensorFlow, scikit-learn, GPU libraries or cloud SDK are installed.
**PyArrow is necessary:** existing shared label/history imports reach
`etl.history_catalog`; retaining it avoids refactoring the completed pipeline.
The build must restore the booster successfully before finishing. UID 10001 runs
the API and cannot write the baked artifacts. Image size is unmeasured until
Phase 3; configuration inspection is not a container-build claim.

Prepare a create-only, small, ignored source context locally (no network):

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m citypulse_ai.api.cloudrun_bundle \
  --output outputs/cloudrun-xgboost/new-context
```

It verifies sources and copied bytes against the catalog, then copies only the
three artifacts and required API/features/forecasting/ETL source directories.
SQLPilot, ExperimentLab, caches, historical outputs and credentials are omitted.
It does not modify original artifacts. Use a new directory if interrupted; do not
submit partial output. The prepared context's `.gcloudignore` does not inherit
the root `.gitignore` that intentionally excludes model files. Submit **only the
prepared context**, never the entire repository. Docker's separate ignore file
adds another allowlist. Contexts/artifacts stay outside Git.

## Terraform ownership and security

`infra/terraform` uses local state only. Terraform 1.17.1 was installed through
Homebrew; the dependency lock selects Google provider 7.46.2.
State, plans, tfvars and `.terraform/` are ignored; preserve local state securely
for future updates/cleanup. No remote backend or CI/CD trigger is added.

| Terraform-owned item | Scope |
| --- | --- |
| Three API declarations | Run, Artifact Registry, Cloud Build; `disable_on_destroy=false` |
| One private Docker repository | `europe-west3/citypulse-forecasting`; immutable tags; optional scanning disabled |
| Runtime service account | `citypulse-xgb-runtime`; **no project roles or GCS permissions** |
| Build service account | `citypulse-xgb-builder`; writer on this repository, project log writer, conditional read on build-source prefix only |
| One Cloud Run service (deployment phase) | `citypulse-xgboost`; invoker IAM check enabled |
| One invoker binding | Named user/service account only; public principals rejected |

The sole existing-bucket IAM addition, if approved/applied, grants the builder
`roles/storage.objectViewer` **conditioned on** objects under
`cloudrun-build-source/`. It cannot list the bucket or read historical/model
objects outside that prefix; there is no object write/delete grant. The operator
stages the small source archive into that prefix. This avoids creating another
source-staging bucket. Historical bucket ownership/public-prevention settings,
existing WIF and the stopped VM/disk are not managed by this Terraform state.
No project-wide storage role, service-account key or public invocation is added.
Existing WIF remains restricted to its original verification workflow; deployment
uses existing operator ADC. The new builder is not the VM/Compute default account.

The service configuration is request-based (`cpu_idle=true`), **0.5 vCPU / 512 MiB**,
first-generation execution, concurrency **1**, min **0** / max **1** at service
and revision level, startup CPU boost off, 60-second request timeout and an HTTP
startup readiness probe. Fractional CPU requires first-generation execution,
request-based billing and concurrency 1. These are initial settings, not a
measured Cloud Run capacity guarantee. Rollout can briefly exceed an instance
limit; max 1 is not a spending cap. Ingress allows external requests but **IAM
requires authentication**; it is not anonymous public access.

## Phase 2 validation and plans

```sh
terraform -chdir=infra/terraform init -input=false
terraform -chdir=infra/terraform fmt -check
terraform -chdir=infra/terraform validate
terraform -chdir=infra/terraform plan -input=false -out=bootstrap.tfplan
```

Default `deploy_service=false` plans bootstrap only: **9 additions, zero changes,
zero deletions**. A full-service preview also succeeded: **11 additions, zero
changes, zero deletions**, with explicit authenticated invoker and an all-zero
**nonexistent placeholder** image digest. The private text plan is
`outputs/cloudrun-xgboost/terraform-review-plan.txt`; the saved binary plan is
`infra/terraform/review-only.tfplan`. **Never apply that review-only plan**.
Cloud Run APIs being disabled did not prevent empty-state planning; remote
quota/service deployment validation remains a Phase 3 prerequisite after approved
API enablement. ADC refresh succeeded; no tokens were printed or saved in reports.
No pre-existing resources were imported or scheduled for deletion.

Focused checks: **47 deployment/shared-contract and XGBoost feature regressions
passed; one native training test was explicitly deselected**. Repository static
checks covered 157 Python files and passed; whitespace/formatting passed. A native TestClient smoke test
loaded the real frozen model, returned healthy + 100 stations + 24 predictions,
and exactly matched the approved saved validation demo (max absolute error 0).
Repeated predictions matched and non-XGBoost selection returned 422. With heavy
framework imports deliberately unavailable, no Torch/TensorFlow/sklearn loaded.
Observed native smoke time was 0.64 s and max process RSS 186,400,768 bytes
(~178 MiB), **not deployed HTTP latency or a container memory measurement**.
The pinned base manifest includes linux/amd64; only public manifest metadata was
read. No local Docker build/container was started. The existing TestClient emits
a Starlette/httpx deprecation warning; no dependency/environment change was made
to hide it.

## Phase 3 — separate approval required

These are future operations, **not executed in Phase 2**:

1. Reconfirm Free Trial/credits and approve a small gross trial-credit budget
   (suggest $1 for this pilot; an estimate, not a hard cap). Never upgrade billing.
   Review Cloud Run spend-cap availability in Billing; regular alerts are not caps.
2. Approve/bootstrap Terraform's API enablement, repository, two identities and
   narrowly scoped IAM. Recheck resource name collisions, regional quotas and
   permissions once APIs are enabled. An existing name must be inspected, not
   overwritten/imported automatically. Review a fresh plan, then apply only the
   approved bootstrap plan. No service is deployed by default.
3. Prepare/inspect a fresh tiny context containing final reviewed source. Approve
   its upload into the existing private bucket's build-source prefix and one
   bounded Cloud Build job. No Cloud Build trigger/pipeline is introduced.

Example **future approved-only** build command (`IMAGE_TAG` must be a new unique
image tag; `CONTEXT` is the reviewed prepared directory):

```sh
gcloud builds submit "$CONTEXT" --config="$CONTEXT/cloudbuild.yaml" \
  --project=citypulse-ai-511315 --region=europe-west3 \
  --gcs-source-staging-dir=gs://citypulse-ai-511315-data/cloudrun-build-source \
  --substitutions="_IMAGE=europe-west3-docker.pkg.dev/citypulse-ai-511315/citypulse-forecasting/xgboost:$IMAGE_TAG"
```

The config uses the dedicated builder, `--platform=linux/amd64`, one e2-medium
worker, Cloud Logging only and a **900-second build timeout**. No build or push
has been performed. Record the pushed digest and inspect actual image size/build
cost. A library/architecture or resource failure stops the pilot; do not retry
repeatedly or add heavy dependencies automatically.

4. Supply the **actual verified image digest** and named invoker in a local ignored
   `.tfvars` file, with `deploy_service=true`. Generate and review a fresh complete
   plan; only after approval apply it. Preserve these inputs thereafter: reverting
   `deploy_service` to false would plan service removal. Do not apply the preview
   digest or reuse a stale plan. API enablement remains a separately approved change.
5. Authenticate invocation and verify real `/health`, `/stations`, `/predict`,
   24 timestamps/counts, station identity, saved-validation parity, deterministic
   response, future-input rejection, logs and cold/warm latency. Do not log tokens
   or whole request histories. The existing validation fixture is
   `outputs/forecasting-v1/demo-request.json`; no test-split evaluation is needed.
6. Only after successful verification review/commit/push the implementation and
   update this guide/README with actual service URL and measurements. No public
   invoker grant is part of this plan; portfolio-public access needs a new decision.

## Costs, retention and cleanup

Request-based Cloud Run still bills startup, requests and shutdown; zero minimum
instances removes idle reservation, not all possible cost. Initial smoke compute
is estimated below $0.10 before allowances. One short e2-medium build is estimated
at a few cents (listed default price $0.003/build-minute); free build allowances
must not be assumed to cover this machine type. Registry storage beyond the
account's first 0.5 GiB is approximately $0.10/GiB-month. Staged source objects,
logs, network egress and existing retained disk/dataset storage can also incur
cost. No scanning or continuously running VM is added. Free-tier credits/limits
are account-wide; no guarantee of zero cost is made.

Cloud Run now supports service-scoped monthly budget spend caps. They use gross
estimated costs, can overshoot before enforcement and do not cap registry,
source/log storage or the retained VM disk. Do not replace existing budgets or
change billing while preparing this code. Reconfirm current Free Trial status and available credits privately before any
billable Phase 3 operation; personal billing balances are not repository configuration.

Keep one initial image/tag. No automatic image/source deletion or prune job is
configured. Before a later replacement, approve removal of an obsolete image only
after confirming no retained revision/rollback depends on it. For future cleanup,
review `terraform plan -destroy` with the same saved variables/state first; only
an explicit cleanup approval permits `terraform destroy`. It affects the new
service, new repo/identities and their own additive IAM bindings. API declarations
leave APIs enabled on destroy. **Never delete the existing dataset bucket, WIF,
VM/disk, model originals or historical objects.** Source staging cleanup, if
wanted later, must target only individually verified newly created source objects.
Preserve state/backups until cleanup/recovery is verified.

References: [fractional CPU requirements](https://docs.cloud.google.com/run/docs/configuring/services/cpu),
[Cloud Run billing](https://docs.cloud.google.com/run/docs/configuring/billing-settings),
[Cloud Run spend-cap limitations](https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps),
[Cloud Build pricing](https://cloud.google.com/build/pricing),
[Artifact Registry pricing](https://cloud.google.com/artifact-registry/pricing).

## CI coverage for this preparation

The existing `.github/workflows/ci.yml` gateway job also runs
`tests/test_cloudrun_xgboost.py`; PyYAML is a test-only dependency for checking
the build configuration. This extends ordinary tests, not deployment automation.
Its push/PR triggers remain unchanged. The new tests use synthetic fixtures and
controlled inference doubles, not packaged weights or real cloud resources.
Terraform apply, GCP API enablement and Cloud Build submission are never invoked
by this CI change. Linux image build/deployed checks remain Phase 3 work.

The preservation check before committing reran 133 lightweight ETL, forecasting
preparation/baseline, causal XGBoost feature and API tests (one actual-training
test deselected), plus 65 SQLPilot and 19 ExperimentLab tests: **217 passed, zero
failed/skipped, one deselected**. The full monolithic suite was not run locally:
it also includes model-training/artifact tests and a separate TensorFlow runtime.
The existing hosted gateway/worker CI jobs provide additional framework checks
without stressing the memory-constrained Mac. One existing Starlette/httpx
deprecation warning was reported; it was not hidden or fixed by unrelated changes.
