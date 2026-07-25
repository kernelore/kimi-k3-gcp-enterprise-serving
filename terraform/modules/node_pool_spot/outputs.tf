output "node_pool_name" {
  description = "GKE Spot GPU Node Pool Name"
  value       = google_container_node_pool.gpu_pool_spot.name
}
