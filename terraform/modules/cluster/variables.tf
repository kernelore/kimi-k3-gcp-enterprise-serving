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

variable "cluster_name" {
  description = "GKE Cluster Name"
  type        = string
}

variable "network_name" {
  description = "VPC Network Name"
  type        = string
}

variable "subnet_name" {
  description = "Subnetwork Name"
  type        = string
}

variable "enable_private_endpoint" {
  description = "Whether to enable private endpoint on the GKE cluster control plane"
  type        = bool
  default     = false
}

variable "master_authorized_cidrs" {
  description = "List of CIDR blocks authorized to access the GKE master endpoint"
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
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

variable "trajectory_bucket_name" {
  description = "Name of the GCS bucket for trajectory logging and audit"
  type        = string
}

variable "weights_cache_bucket_name" {
  description = "Name of the GCS bucket for persistent weights cache"
  type        = string
}
