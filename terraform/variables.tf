variable "project_id" {
  description = "GCP Project ID where the Kimi K3 infrastructure will be deployed"
  type        = string
}

variable "region" {
  description = "GCP Region for Sovereign Deployment (e.g., europe-north1)"
  type        = string
  default     = "europe-north1"
}

variable "zone" {
  description = "GCP Zone for Compact Placement and GPU Nodes (e.g., europe-north1-b for B200)"
  type        = string
  default     = "europe-north1-b"
}

variable "cluster_name" {
  description = "GKE Cluster Name"
  type        = string
  default     = "kimi-enterprise-fi"
}

variable "gpu_machine_type" {
  description = "GPU Machine Type for Serving Pool (e.g., a4-highgpu-8g with 8x NVIDIA B200)"
  type        = string
  default     = "a4-highgpu-8g"
}

variable "model_family" {
  description = "Target AI model family deployed on the cluster"
  type        = string
  default     = "kimi-k3"
  validation {
    condition     = can(regex("^[a-z0-9]+(-[a-z0-9]+)*$", var.model_family)) && var.model_family == "kimi-k3"
    error_message = "The model_family variable must be formatted in kebab-case and explicitly set to 'kimi-k3'."
  }
}

variable "hyperdisk_ml_size_gb" {
  description = "Persistent size in GB for the shared Hyperdisk ML volume (2000 for Kimi K3)"
  type        = number
  default     = 2000
}

variable "hyperdisk_ml_throughput_mibps" {
  description = "Provisioned throughput (MiB/s) for the Hyperdisk ML weights volume. Volume max is MIN(2097152, 1600 * size_gib). The binding limit is the regional HDML_TOTAL_THROUGHPUT quota (30720 MiB/s), so 24576 leaves headroom for one concurrent stack. Throughput is shared across all attached instances and can only be modified once every 6 hours, so it must be chosen at creation time. If quota is unavailable, 6144 is a known-good fallback."
  type        = number
  default     = 24576
}

variable "nodes_per_replica" {
  description = "Number of GKE Blackwell nodes required per serving pod replica (2 nodes MVP for Kimi K3)"
  type        = number
  default     = 2
}

variable "owner_label" {
  description = "Mandatory Owner Label (must match ^[a-z0-9-_]+$)"
  type        = string
  default     = "opensource-user"
}

variable "ttl_label" {
  description = "Mandatory TTL Label (e.g., 7d, 24h)"
  type        = string
  default     = "7d"
}

variable "env_label" {
  description = "Mandatory Environment Label"
  type        = string
  default     = "kimi-k3-prod"
}

variable "db_tier" {
  description = "Cloud SQL machine tier for the gateway PostgreSQL instance"
  type        = string
  default     = "db-custom-4-16384"
}

variable "db_password" {
  description = "Password for the Cloud SQL gateway_admin user"
  type        = string
  sensitive   = true
  validation {
    condition     = length(var.db_password) >= 8 && var.db_password != "ChangeMeEnterpriseProd123!" && var.db_password != "REPLACE_WITH_SECURE_PASSWORD"
    error_message = "The db_password variable must be provided, at least 8 characters long, and not use a default placeholder."
  }
}

variable "enable_private_endpoint" {
  description = "Whether to enable private IP endpoint for GKE Control Plane access"
  type        = bool
  default     = false
}

variable "master_authorized_cidrs" {
  description = "List of master authorized CIDR blocks for GKE cluster access"
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}

variable "gpu_pool_max_nodes" {
  description = "Maximum number of nodes in spot GPU pool"
  type        = number
  default     = 2
}

