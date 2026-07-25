variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region for Cloud SQL instance"
  type        = string
}

variable "network_id" {
  description = "Authorized VPC network ID for private services access peering"
  type        = string
}

variable "db_tier" {
  description = "Cloud SQL machine tier for the gateway PostgreSQL instance"
  type        = string
}

variable "db_password" {
  description = "Password for the Cloud SQL gateway_admin user"
  type        = string
  sensitive   = true
}

variable "owner_label" {
  description = "Mandatory Owner Label"
  type        = string
}

variable "ttl_label" {
  description = "Mandatory TTL Label"
  type        = string
}

variable "env_label" {
  description = "Mandatory Environment Label"
  type        = string
}
