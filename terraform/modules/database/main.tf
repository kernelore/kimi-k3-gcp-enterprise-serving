resource "google_compute_global_address" "private_ip_alloc" {
  # Pin PSA range explicitly. Without this attribute Service Networking
  # auto-selects it and can collide with the auto-allocated GKE pod/service
  # ranges in modules/cluster (the two allocators do not coordinate;
  # the collision is an ordering race on fresh deploys).
  name          = "kimi-k3-gateway-psa-range"
  project       = var.project_id
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  address       = "10.90.0.0"
  prefix_length = 16
  network       = var.network_id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = var.network_id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_alloc.name]
  deletion_policy         = "ABANDON"
}

resource "google_sql_database_instance" "gateway_db" {
  name             = "kimi-k3-gateway-db"
  database_version = "POSTGRES_15"
  region           = var.region
  project          = var.project_id

  deletion_protection = false

  settings {
    tier = var.db_tier

    ip_configuration {
      ipv4_enabled    = false
      private_network = var.network_id
    }

    user_labels = {
      env   = var.env_label
      owner = var.owner_label
      ttl   = var.ttl_label
    }
  }

  depends_on = [google_service_networking_connection.private_vpc_connection]
}

resource "google_sql_database" "default" {
  name     = "kimi_k3_gateway"
  instance = google_sql_database_instance.gateway_db.name
  project  = var.project_id
}

resource "google_sql_user" "gateway_user" {
  name            = "gateway_admin"
  instance        = google_sql_database_instance.gateway_db.name
  project         = var.project_id
  password        = var.db_password
  deletion_policy = "ABANDON"
  depends_on      = [google_sql_database.default]
}
