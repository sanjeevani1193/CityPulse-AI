# Proposed Phase 4 grants: no apply until the reviewed plan is approved.
# A separate pool leaves the existing verification provider entirely untouched.
resource "google_iam_workload_identity_pool" "cd" {
  project                   = var.project_id
  workload_identity_pool_id = "citypulse-cloudrun-cd"
  display_name              = "CityPulse CD WIF"
}

resource "google_iam_workload_identity_pool_provider" "cd" {
  project                            = var.project_id
  workload_identity_pool_id          = google_iam_workload_identity_pool.cd.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-main"
  attribute_mapping = {
    "google.subject"          = "assertion.sub"
    "attribute.repository_id" = "assertion.repository_id"
  }
  attribute_condition = "assertion.repository_id == '1410625891' && assertion.repository_owner_id == '91110975' && assertion.ref == 'refs/heads/main' && assertion.workflow_ref == 'sanjeevani1193/CityPulse-AI/.github/workflows/cloudrun-cd.yml@refs/heads/main' && assertion.event_name in ['push', 'workflow_dispatch']"
  oidc { issuer_uri = "https://token.actions.githubusercontent.com" }
}

resource "google_service_account" "cd" {
  project      = var.project_id
  account_id   = "citypulse-xgb-deployer"
  display_name = "CityPulse trusted main deployment"
}

resource "google_service_account_iam_member" "cd_federation" {
  service_account_id = google_service_account.cd.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.cd.name}/attribute.repository_id/1410625891"
}

resource "google_project_iam_custom_role" "cd_build" {
  project     = var.project_id
  role_id     = "citypulseXgbBuildSubmit"
  title       = "CityPulse bounded build submission"
  permissions = ["cloudbuild.builds.create", "cloudbuild.builds.get", "serviceusage.services.use"]
}

resource "google_project_iam_member" "cd_build" {
  project = var.project_id
  role    = google_project_iam_custom_role.cd_build.name
  member  = "serviceAccount:${google_service_account.cd.email}"
}

# actAs is restricted to these two existing identities, not project-wide.
resource "google_service_account_iam_member" "cd_act_as" {
  for_each           = { builder = google_service_account.builder.name, runtime = google_service_account.runtime.name }
  service_account_id = each.value
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.cd.email}"
}

resource "google_project_iam_custom_role" "cd_run" {
  project     = var.project_id
  role_id     = "citypulseXgbServiceUpdate"
  title       = "CityPulse existing service update"
  permissions = ["run.services.get", "run.services.update"]
}

resource "google_cloud_run_v2_service_iam_member" "cd_update" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api[0].name
  role     = google_project_iam_custom_role.cd_run.name
  member   = "serviceAccount:${google_service_account.cd.email}"
}

resource "google_cloud_run_v2_service_iam_member" "cd_invoke" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api[0].name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.cd.email}"
}

resource "google_artifact_registry_repository_iam_member" "cd_reader" {
  project    = var.project_id
  location   = var.region
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${google_service_account.cd.email}"
}

resource "google_storage_bucket_iam_member" "cd_inputs" {
  bucket = "citypulse-ai-511315-data"
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cd.email}"
  condition {
    title      = "citypulse_cd_exact_frozen_inputs"
    expression = "resource.name in ${jsonencode([for item in jsondecode(file("${path.module}/../cloudrun-cd-inputs.json")).inputs : "projects/_/buckets/citypulse-ai-511315-data/objects/${split("citypulse-ai-511315-data/", item.cloud_path)[1]}"])}"
  }
}

resource "google_storage_bucket_iam_member" "cd_source_create" {
  bucket = "citypulse-ai-511315-data"
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.cd.email}"
  condition {
    title      = "citypulse_cd_create_build_source_only"
    expression = "resource.name.startsWith('projects/_/buckets/citypulse-ai-511315-data/objects/cloudrun-build-source/cd/')"
  }
}

output "cd_provider" { value = google_iam_workload_identity_pool_provider.cd.name }
output "cd_service_account" { value = google_service_account.cd.email }
