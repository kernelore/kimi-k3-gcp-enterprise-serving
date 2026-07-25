output "redis_host" {
  description = "Cloud Memorystore Redis host IP address"
  value       = google_redis_instance.cache.host
}

output "redis_port" {
  description = "Cloud Memorystore Redis port number"
  value       = google_redis_instance.cache.port
}

output "redis_auth_string" {
  description = "Cloud Memorystore Redis auth string"
  value       = google_redis_instance.cache.auth_string
  sensitive   = true
}
