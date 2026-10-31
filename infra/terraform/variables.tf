variable "project_id" {
  type    = string
  default = "citypulse-ai-511315"
  validation {
    condition     = var.project_id == "citypulse-ai-511315"
    error_message = "This demo is restricted to the existing CityPulse project."
  }
}
variable "region" {
  type    = string
  default = "europe-west3"
  validation {
    condition     = var.region == "europe-west3"
    error_message = "Keep deployment in Frankfurt."
  }
}
variable "deploy_service" {
  type        = bool
  default     = false
  description = "Bootstrap only by default; enable only after the image exists and a deployment plan is approved."
}
variable "image_digest" {
  type        = string
  default     = ""
  description = "Verified pushed sha256 digest, never a mutable tag."
  validation {
    condition     = var.image_digest == "" || can(regex("^sha256:[0-9a-f]{64}$", var.image_digest))
    error_message = "Use an immutable sha256 image digest."
  }
}
variable "invoker_member" {
  type        = string
  default     = ""
  description = "One approved user:email or serviceAccount:email. Never a public principal."
  validation {
    condition     = var.invoker_member == "" || can(regex("^(user|serviceAccount):[^[:space:]@]+@[^[:space:]@]+$", var.invoker_member))
    error_message = "Only a named user or service account is permitted; public access is forbidden."
  }
}
