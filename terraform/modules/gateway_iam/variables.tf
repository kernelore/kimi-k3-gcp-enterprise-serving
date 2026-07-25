variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "dataset_id" {
  description = "BigQuery dataset ID for audit log data editor permissions"
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
