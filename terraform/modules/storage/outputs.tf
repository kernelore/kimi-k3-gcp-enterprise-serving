output "staging_disk_name" {
  value       = google_compute_disk.staging_disk.name
  description = "Name of the Hyperdisk ML staging disk"
}

output "trajectory_bucket_name" {
  value       = google_storage_bucket.trajectory_bucket.name
  description = "Name of the trajectory storage bucket"
}

output "weights_cache_bucket_name" {
  value       = google_storage_bucket.weights_cache.name
  description = "Name of the weights cache storage bucket"
}

output "hyperdisk_ml_rox_id" {
  value       = "projects/${var.project_id}/zones/${var.zone}/disks/${google_compute_disk.staging_disk.name}"
  description = "Resource ID of the 2,000 GB Hyperdisk ML ROX volume"
}

output "artifact_registry_repo" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.kimi_repo.repository_id}"
  description = "Docker repository path for Kimi K3 containers"
}
