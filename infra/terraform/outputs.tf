output "repository" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}"
}
output "runtime_service_account" {
  value = google_service_account.runtime.email
}
output "builder_service_account" {
  value = google_service_account.builder.email
}
output "service_url" {
  value = var.deploy_service ? google_cloud_run_v2_service.api[0].uri : null
}
