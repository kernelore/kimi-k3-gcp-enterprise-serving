# 2,000 GB Persistent Hyperdisk ML Shared Staging & Serving Volume for Kimi K3 MXFP4
resource "google_compute_disk" "staging_disk" {
  name                   = "kimi-k3-weights-rox"
  type                   = "hyperdisk-ml"
  size                   = var.hyperdisk_ml_size_gb
  provisioned_throughput = var.hyperdisk_ml_throughput_mibps
  zone                   = var.zone
  project                = var.project_id

  labels = {
    env   = var.env_label
    owner = var.owner_label
    ttl   = var.ttl_label
    model = var.model_family
  }

  lifecycle {
    ignore_changes = [access_mode]
  }
}

# Trajectory and Logging Bucket (7-day lifecycle)
resource "google_storage_bucket" "trajectory_bucket" {
  name                        = "${var.project_id}-${var.model_family}-trajectories"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = true
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }

  labels = {
    env   = var.env_label
    owner = var.owner_label
    ttl   = var.ttl_label
  }
}

# Persistent Weights Cache Bucket (force_destroy=true ensures complete teardown on terraform destroy)
resource "google_storage_bucket" "weights_cache" {
  name                        = "${var.project_id}-${var.model_family}-weights-cache"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = true
  uniform_bucket_level_access = true

  labels = {
    env   = var.env_label
    owner = var.owner_label
    ttl   = var.ttl_label
  }
}

# Sovereign Artifact Registry Repository for Kimi K3 Container Images
resource "google_artifact_registry_repository" "kimi_repo" {
  location      = var.region
  repository_id = "kimi-prod"
  description   = "Docker container repository for Kimi K3 dual-engine (TensorRT-LLM / SGLang) inference"
  format        = "DOCKER"
  project       = var.project_id

  labels = {
    env   = var.env_label
    owner = var.owner_label
    ttl   = var.ttl_label
  }
}
