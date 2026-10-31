terraform {
  required_version = ">= 1.9, < 2.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }
  # Intentionally local state. Never add the historical bucket as a backend.
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "deployment" {
  for_each           = toset(["run.googleapis.com", "artifactregistry.googleapis.com", "cloudbuild.googleapis.com"])
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  project       = var.project_id
  location      = var.region
  repository_id = "citypulse-forecasting"
  description   = "Private XGBoost-only deployment images; no historical data"
  format        = "DOCKER"
  docker_config {
    immutable_tags = true
  }
  vulnerability_scanning_config {
    enablement_config = "DISABLED"
  }
  depends_on = [google_project_service.deployment]
}

resource "google_service_account" "runtime" {
  project      = var.project_id
  account_id   = "citypulse-xgb-runtime"
  display_name = "CityPulse XGBoost runtime — no project roles"
}

resource "google_service_account" "builder" {
  project      = var.project_id
  account_id   = "citypulse-xgb-builder"
  display_name = "CityPulse manual image builder"
}

resource "google_artifact_registry_repository_iam_member" "builder_writer" {
  project    = var.project_id
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.builder.email}"
}

resource "google_project_iam_member" "builder_logs" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.builder.email}"
}

# Additive conditional read grant, never manage/delete the existing bucket itself.
# No list/delete/write access to historical data or model objects.
resource "google_storage_bucket_iam_member" "builder_source" {
  bucket = "citypulse-ai-511315-data"
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.builder.email}"
  condition {
    title       = "citypulse_xgb_build_source_only"
    description = "Only manually staged source archives for this deployment"
    expression  = "resource.name.startsWith('projects/_/buckets/citypulse-ai-511315-data/objects/cloudrun-build-source/')"
  }
}

resource "google_cloud_run_v2_service" "api" {
  count                = var.deploy_service ? 1 : 0
  project              = var.project_id
  name                 = "citypulse-xgboost"
  location             = var.region
  ingress              = "INGRESS_TRAFFIC_ALL"
  invoker_iam_disabled = false
  deletion_protection  = false # Demo cleanup affects only this newly owned service.
  scaling {
    min_instance_count = 0
    max_instance_count = 1
  }
  template {
    service_account                  = google_service_account.runtime.email
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN1"
    max_instance_request_concurrency = 1
    timeout                          = "60s"
    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/xgboost@${var.image_digest}"
      ports {
        container_port = 8080
      }
      resources {
        limits = {
          cpu    = "0.5"
          memory = "512Mi"
        }
        cpu_idle          = true
        startup_cpu_boost = false
      }
      startup_probe {
        initial_delay_seconds = 0
        timeout_seconds       = 3
        period_seconds        = 10
        failure_threshold     = 18
        http_get {
          path = "/health"
          port = 8080
        }
      }
    }
  }
  lifecycle {
    # CD owns only the image. Terraform still detects all other service drift.
    ignore_changes = [template[0].containers[0].image]
    precondition {
      condition     = var.image_digest != "" && var.invoker_member != ""
      error_message = "Deployment requires a verified image digest and an explicit authenticated invoker."
    }
  }
  depends_on = [google_project_service.deployment]
}

resource "google_cloud_run_v2_service_iam_member" "invoker" {
  count    = var.deploy_service ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api[0].name
  role     = "roles/run.invoker"
  member   = var.invoker_member
}
