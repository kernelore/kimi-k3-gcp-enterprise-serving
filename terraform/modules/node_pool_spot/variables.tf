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

variable "gpu_machine_type" {
  description = "GPU Machine Type for Serving Pool"
  type        = string
}

variable "nodes_per_replica" {
  description = "Number of GKE Blackwell nodes required per serving pod replica"
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

variable "roce_network_name" {
  description = "RoCEv2 secondary VPC network name"
  type        = string
}

variable "roce_subnetwork_names" {
  description = "List of RoCEv2 secondary subnetwork names"
  type        = list(string)
}

variable "gpu_pool_max_nodes" {
  description = "Maximum number of nodes in spot GPU pool"
  type        = number
  default     = 2
}

variable "secondary_gvnic_network_name" {
  description = "Secondary VPC network name for GVNIC interface (NIC 1)"
  type        = string
  default     = "gvnic-net"
}

variable "secondary_gvnic_subnetwork_name" {
  description = "Secondary subnetwork name for GVNIC interface (NIC 1)"
  type        = string
  default     = "gvnic-sub"
}

