variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region for Redis instance"
  type        = string
}

variable "network_id" {
  description = "Authorized VPC network ID for Redis private connection"
  type        = string
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
