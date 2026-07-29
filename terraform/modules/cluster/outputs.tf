output "cluster_name" {
  description = "GKE Cluster Name"
  value       = google_container_cluster.primary.name
}

output "endpoint" {
  description = "GKE Cluster Control Plane Endpoint"
  value       = google_container_cluster.primary.endpoint
}

output "serving_service_account_email" {
  description = "Workload Identity Service Account Email for Serving Workloads"
  value       = google_service_account.serving_sa.email
}

output "workload_identity_sa" {
  description = "Workload Identity Service Account Email for Serving Workloads (alias)"
  value       = google_service_account.serving_sa.email
}

output "node_service_account_email" {
  description = "Service Account Email for GKE Node Pools"
  value       = google_service_account.node_sa.email
}
