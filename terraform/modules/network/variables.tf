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

variable "vpc_name" {
  description = "Primary VPC Network Name"
  type        = string
  default     = "kimi-k3-vpc"
}

variable "subnet_name" {
  description = "Primary Subnet Name"
  type        = string
  default     = "kimi-k3-subnet"
}

variable "subnet_cidr" {
  description = "Primary Subnetwork CIDR"
  type        = string
  default     = "10.10.0.0/16"
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
