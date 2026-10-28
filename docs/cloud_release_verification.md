# Frozen station100 verification on GitHub-hosted ARM64

Status: workflow prepared; **no verification success is claimed until its run
passes**. Local Mac memory pressure prevents live services. This moves only
verification to a standard GitHub-hosted Ubuntu ARM64 runner, not Compute Engine.
No training, historical ingestion or held-out inference is part of this workflow.

## Exact cloud trust configured October 28, 2026

Project `citypulse-ai-511315`, number `64394592274`:

- Enabled **only** `sts.googleapis.com`; IAM and IAM Credentials were already enabled.
- Pool: `projects/64394592274/locations/global/workloadIdentityPools/citypulse-github-verification`.
- OIDC provider: the pool's `providers/github-manual`, issuer
  `https://token.actions.githubusercontent.com`.
- Mappings: `google.subject=assertion.sub`,
  `attribute.repository_id=assertion.repository_id`,
  `attribute.repository_owner_id=assertion.repository_owner_id`.
- Actual provider condition requires **all** of:
  repository ID `1410625891`, owner ID `91110975`, ref `refs/heads/main`,
  event `workflow_dispatch`, workflow reference
  `sanjeevani1193/CityPulse-AI/.github/workflows/forecasting-verification.yml@refs/heads/main`.
- Direct federated principal:
  `principalSet://iam.googleapis.com/projects/64394592274/locations/global/workloadIdentityPools/citypulse-github-verification/attribute.repository_id/1410625891`.
- Custom role: `projects/citypulse-ai-511315/roles/citypulseForecastVerificationRead`,
  GA, with **only `storage.objects.get`**.
- The sole new bucket binding uses that principal and role, conditioned on the
  **eleven exact object names** in [the recorded condition](forecasting_verification_iam_condition.json).
  There is no object listing, write, delete or administrative permission.

There is no service account, impersonation grant or permanent key for this workflow.
Existing pilot IAM bindings are preserved. Bucket uniform access and enforced
public-access prevention were already enabled and remain unchanged. No billing
account settings, compute resources or bucket contents were changed.

IAM restricts object **names**; it does not pin generations. The downloader separately
requires the catalog's exact generations, metadata sizes, SHA-256 values and member
checksums. The allowlist contains the seven small serving files, three saved
validation prediction files, and the existing dataset recovery object. Only one
22.27 MB validation-panel member is read by ranges from the recovery ZIP; the
historical dataset ZIP is never downloaded. Total intended reads are tens of MB,
plus ZIP seek overhead, within the approved small trial-credit transfer budget.
No test-quarter predictions are downloaded by this workflow.

## Workflow and privacy

`.github/workflows/forecasting-verification.yml` is manual-only. Its single job
needs `contents: read` and `id-token: write` for short-lived federation. GCP enforces
repository/owner/branch/event/workflow restrictions independently of GitHub job
conditions. Ordinary CI and fork/PR jobs cannot authenticate using this provider.
Security-sensitive actions are pinned to immutable upstream commits.

The job checks capacity before installing/building, isolates Python 3.14 gateway
and Python 3.11 TensorFlow environments, restores pinned artifacts, runs synthetic
framework regressions and real native HTTP parity, then builds images sequentially
and tests Compose. The job timeout is 75 minutes. Native services are terminated in
`finally`; Compose has an `always()` cleanup step and the hosted runner is ephemeral.
The inference monitor requires zero swap use, at least 2 GiB host available memory,
3 GiB free disk and at most 2 GiB combined native child RSS. Compose retains both
1536 MiB caps and CPU limits. Observed resource statistics are not peak-memory
proofs between sampling intervals.

Only `verification-reports/` is uploaded, retained 14 days: aggregate parity/error
summaries, resource numbers and synthetic-test JUnit results. Checkpoints, ADC files,
private demo inputs, raw predictions and private service logs remain in the ephemeral
workspace and are never uploaded or cached. The existing Docker allowlist excludes
credentials and artifacts from image build contexts; models are bind-mounted read-only.

Trigger from GitHub Actions → **Frozen station100 release verification** → Run workflow
on `main`. A source commit/push is necessary before a hosted runner can execute the
exact main-branch workflow trusted by GCP. Such a preparation commit is not a release
completion claim. After successful verification, record its URL and measured results
before marking Forecasting v1 complete.

## Revoke access (instructions only; not performed automatically)

Immediately disable the dedicated provider:

```sh
gcloud iam workload-identity-pools providers update-oidc github-manual \
  --workload-identity-pool=citypulse-github-verification --location=global \
  --project=citypulse-ai-511315 --disabled
```

Then remove **only** the conditional verification binding:

```sh
gcloud storage buckets remove-iam-policy-binding gs://citypulse-ai-511315-data \
  --member='principalSet://iam.googleapis.com/projects/64394592274/locations/global/workloadIdentityPools/citypulse-github-verification/attribute.repository_id/1410625891' \
  --role=projects/citypulse-ai-511315/roles/citypulseForecastVerificationRead \
  --condition-from-file=docs/forecasting_verification_iam_condition.json
```

After confirming nothing else uses this dedicated setup, the pool/provider and
custom role can be deleted separately with approval. Do not disable shared IAM
APIs or remove pilot bindings. Disabling trust blocks new exchanges; previously
issued short-lived tokens may remain valid until expiry. None of these revocation
steps deletes GCS data.
