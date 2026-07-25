output "network_id" {
  description = "VPC Network ID"
  value       = google_compute_network.vpc.id
}

output "network_name" {
  description = "VPC Network Name"
  value       = google_compute_network.vpc.name
}

output "primary_subnet_name" {
  description = "Primary Subnetwork Name"
  value       = google_compute_subnetwork.primary_subnet.name
}

output "roce_network_name" {
  description = "RoCEv2 Secondary VPC Network Name"
  value       = google_compute_network.roce_network.name
}

output "roce_subnetwork_names" {
  description = "RoCEv2 Secondary Subnetwork Names"
  value       = [for s in google_compute_subnetwork.roce_subnet : s.name]
}

output "gvnic_network_name" {
  description = "GVNIC Secondary VPC Network Name"
  value       = google_compute_network.gvnic_network.name
}

output "gvnic_subnetwork_name" {
  description = "GVNIC Secondary Subnetwork Name"
  value       = google_compute_subnetwork.gvnic_subnet.name
}
