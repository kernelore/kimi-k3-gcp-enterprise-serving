output "db_instance_connection_name" {
  description = "Cloud SQL database instance connection name for Cloud SQL Auth Proxy"
  value       = google_sql_database_instance.gateway_db.connection_name
}

output "db_host" {
  description = "Cloud SQL database private IP host address"
  value       = google_sql_database_instance.gateway_db.private_ip_address
}

output "db_user" {
  description = "Cloud SQL database username"
  value       = google_sql_user.gateway_user.name
}
