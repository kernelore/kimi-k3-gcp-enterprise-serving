output "gateway_service_account_email" {
  description = "Workload Identity Service Account Email for the Enterprise AI Gateway"
  value       = google_service_account.gateway_sa.email
}
