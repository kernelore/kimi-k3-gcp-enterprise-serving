variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}

variable "zone" {
  description = "GCP Zone"
  type        = string
}

variable "model_family" {
  description = "Target AI model family"
  type        = string
  validation {
    condition     = can(regex("^[a-z0-9]+(-[a-z0-9]+)*$", var.model_family)) && var.model_family == "kimi-k3"
    error_message = "The model_family variable must be formatted in kebab-case and explicitly set to 'kimi-k3'."
  }
}

variable "hyperdisk_ml_size_gb" {
  description = "Persistent size in GB for the shared Hyperdisk ML volume"
  type        = number
}

variable "hyperdisk_ml_throughput_mibps" {
  description = "Provisioned throughput in MiB/s for Hyperdisk ML staging volume"
  type        = number
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
