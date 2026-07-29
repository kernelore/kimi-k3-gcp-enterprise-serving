resource "google_compute_resource_policy" "compact_placement" {
  name    = "pp-kimi-b200-roce"
  region  = var.region
  project = var.project_id
  group_placement_policy {
    collocation = "COLLOCATED"
  }
}

resource "google_container_node_pool" "gpu_pool_spot" {
  provider = google-beta
  name     = "kimi-k3-b200-spot"
  location = var.zone
  cluster  = var.cluster_name
  project  = var.project_id
  autoscaling {
    total_min_node_count = 0
    total_max_node_count = var.gpu_pool_max_nodes
  }

  placement_policy {
    type        = "COMPACT"
    policy_name = google_compute_resource_policy.compact_placement.name
  }

  node_config {
    machine_type    = var.gpu_machine_type
    service_account = var.node_service_account_email
    spot            = true
    disk_size_gb    = 200

    # gVNIC is default/mandatory on A4 primary NIC; setting it explicitly forces GVNIC onto RDMA NICs and is rejected by the RoCE network profile (allowed: MRDMA).

    gcfs_config {
      enabled = true
    }

    local_nvme_ssd_block_config {
      local_ssd_count = 32
    }

    guest_accelerator {
      type  = "nvidia-b200"
      count = 8
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    labels = {
      env   = var.env_label
      owner = var.owner_label
      ttl   = var.ttl_label
      model = "kimi-k3"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  network_config {
    enable_private_nodes = true # pool-level network_config overrides cluster privacy; must be re-asserted here
    # NIC 1: Required secondary GVNIC interface for A4 / A3 Ultra 10-NIC architecture
    additional_node_network_configs {
      network    = var.secondary_gvnic_network_name
      subnetwork = var.secondary_gvnic_subnetwork_name
    }

    # NICs 2-9: 8 RDMA interfaces for GPU-to-GPU RoCE communication
    dynamic "additional_node_network_configs" {
      for_each = range(8)
      content {
        network    = var.roce_network_name
        subnetwork = var.roce_subnetwork_names[additional_node_network_configs.value]
      }
    }
  }

  lifecycle {
    ignore_changes = [node_count, initial_node_count]
  }
}
