terraform {
  required_version = ">= 1.5.0"

  backend "gcs" {
    prefix = "terraform/state/kimi-k3-mxfp4"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.41"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 7.41"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

module "network" {
  source      = "./modules/network"
  project_id  = var.project_id
  region      = var.region
  zone        = var.zone
  owner_label = var.owner_label
  ttl_label   = var.ttl_label
  env_label   = var.env_label
}

module "cluster" {
  source                  = "./modules/cluster"
  project_id              = var.project_id
  region                  = var.region
  zone                    = var.zone
  cluster_name            = var.cluster_name
  network_name            = module.network.network_name
  subnet_name             = module.network.primary_subnet_name
  enable_private_endpoint = var.enable_private_endpoint
  master_authorized_cidrs = var.master_authorized_cidrs
  owner_label             = var.owner_label
  ttl_label               = var.ttl_label
  env_label               = var.env_label
  trajectory_bucket_name  = module.storage.trajectory_bucket_name
}

module "storage" {
  source                        = "./modules/storage"
  project_id                    = var.project_id
  region                        = var.region
  zone                          = var.zone
  model_family                  = var.model_family
  hyperdisk_ml_size_gb          = var.hyperdisk_ml_size_gb
  hyperdisk_ml_throughput_mibps = var.hyperdisk_ml_throughput_mibps
  owner_label                   = var.owner_label
  ttl_label                     = var.ttl_label
  env_label                     = var.env_label
}

module "cache" {
  source      = "./modules/cache"
  project_id  = var.project_id
  region      = var.region
  network_id  = module.network.network_id
  owner_label = var.owner_label
  ttl_label   = var.ttl_label
  env_label   = var.env_label
  depends_on  = [module.network]
}

module "database" {
  source      = "./modules/database"
  project_id  = var.project_id
  region      = var.region
  network_id  = module.network.network_id
  db_tier     = var.db_tier
  db_password = var.db_password
  owner_label = var.owner_label
  ttl_label   = var.ttl_label
  env_label   = var.env_label
  depends_on  = [module.network, module.cache]
}

module "audit" {
  source      = "./modules/audit"
  project_id  = var.project_id
  region      = var.region
  owner_label = var.owner_label
  ttl_label   = var.ttl_label
  env_label   = var.env_label
}

module "gateway_iam" {
  source      = "./modules/gateway_iam"
  project_id  = var.project_id
  dataset_id  = module.audit.dataset_id
  owner_label = var.owner_label
  ttl_label   = var.ttl_label
  env_label   = var.env_label
}

module "observability" {
  source       = "./modules/observability"
  project_id   = var.project_id
  region       = var.region
  cluster_name = module.cluster.cluster_name
  owner_label  = var.owner_label
  ttl_label    = var.ttl_label
  env_label    = var.env_label
}

module "node_pool_spot" {
  source                          = "./modules/node_pool_spot"
  project_id                      = var.project_id
  region                          = var.region
  zone                            = var.zone
  cluster_name                    = module.cluster.cluster_name
  gpu_machine_type                = var.gpu_machine_type
  nodes_per_replica               = var.nodes_per_replica
  gpu_pool_max_nodes              = var.gpu_pool_max_nodes
  secondary_gvnic_network_name    = module.network.gvnic_network_name
  secondary_gvnic_subnetwork_name = module.network.gvnic_subnetwork_name
  roce_network_name               = module.network.roce_network_name
  roce_subnetwork_names           = module.network.roce_subnetwork_names
  node_service_account_email      = module.cluster.node_service_account_email
  owner_label                     = var.owner_label
  ttl_label                       = var.ttl_label
  env_label                       = var.env_label
  depends_on                      = [module.cluster]
}
