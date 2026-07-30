resource "google_container_cluster" "primary" {
  provider   = google-beta
  name       = var.cluster_name
  location   = var.zone
  project    = var.project_id
  network    = var.network_name
  subnetwork = var.subnet_name

  deletion_protection = false

  datapath_provider       = "ADVANCED_DATAPATH"
  enable_multi_networking = true
  release_channel {
    channel = "RAPID"
  }

  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    gcs_fuse_csi_driver_config {
      enabled = true
    }
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = var.enable_private_endpoint
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  # Pin VPC-native pod/service ranges explicitly. Without this block GKE
  # auto-selects them and can collide with the auto-allocated Service
  # Networking PSA range in modules/database (the two allocators do not
  # coordinate; the collision is an ordering race on fresh deploys).
  ip_allocation_policy {
    cluster_ipv4_cidr_block  = "10.64.0.0/14"
    services_ipv4_cidr_block = "10.80.0.0/20"
  }

  dynamic "master_authorized_networks_config" {
    for_each = length(var.master_authorized_cidrs) > 0 ? [1] : []
    content {
      dynamic "cidr_blocks" {
        for_each = var.master_authorized_cidrs
        content {
          cidr_block   = cidr_blocks.value.cidr_block
          display_name = cidr_blocks.value.display_name
        }
      }
    }
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "STORAGE",
      "APISERVER",
      "SCHEDULER",
      "CONTROLLER_MANAGER",
      "POD",
      "DEPLOYMENT",
      "DAEMONSET",
      "STATEFULSET",
      "HPA",
      "CADVISOR",
      "KUBELET",
      "DCGM"
    ]
    managed_prometheus {
      enabled = true
    }
    advanced_datapath_observability_config {
      enable_metrics = true
      enable_relay   = false
    }
  }

  logging_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "APISERVER",
      "SCHEDULER",
      "CONTROLLER_MANAGER"
    ]
  }

  resource_labels = {
    env   = var.env_label
    owner = var.owner_label
    ttl   = var.ttl_label
  }
}

resource "google_service_account" "serving_sa" {
  account_id   = "kimi-k3-serving-sa"
  display_name = "Kimi K3 Serving Workload Identity SA"
  project      = var.project_id
}

resource "google_storage_bucket_iam_member" "trajectory_writer" {
  bucket = var.trajectory_bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.serving_sa.email}"
}

resource "google_storage_bucket_iam_member" "weights_cache_writer" {
  bucket = var.weights_cache_bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.serving_sa.email}"
}

resource "google_project_iam_member" "artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.serving_sa.email}"
}

resource "google_service_account_iam_member" "workload_identity_user" {
  service_account_id = google_service_account.serving_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[llm-serving/kimi-k3-serving-sa]"
}

resource "google_service_account" "node_sa" {
  account_id   = "kimi-k3-node-sa"
  display_name = "Kimi K3 GKE Node Service Account"
  project      = var.project_id
}

resource "google_project_iam_member" "node_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.node_sa.email}"
}

resource "google_project_iam_member" "node_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.node_sa.email}"
}

resource "google_project_iam_member" "node_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.node_sa.email}"
}

resource "google_project_iam_member" "node_monitoring_viewer" {
  project = var.project_id
  role    = "roles/monitoring.viewer"
  member  = "serviceAccount:${google_service_account.node_sa.email}"
}

resource "google_container_node_pool" "system_pool" {
  name       = "np-system"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = 1

  autoscaling {
    min_node_count = 1
    max_node_count = 3
  }

  node_config {
    machine_type    = "e2-standard-8"
    service_account = google_service_account.node_sa.email
    disk_size_gb    = 100

    labels = {
      env   = var.env_label
      owner = var.owner_label
      ttl   = var.ttl_label
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  lifecycle {
    ignore_changes = [node_count]
  }
}
