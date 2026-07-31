resource "google_compute_network" "vpc" {
  name                    = var.vpc_name
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
  mtu                     = 1500
  project                 = var.project_id
}

resource "google_compute_subnetwork" "primary_subnet" {
  name                     = var.subnet_name
  ip_cidr_range            = var.subnet_cidr
  region                   = var.region
  network                  = google_compute_network.vpc.id
  private_ip_google_access = true
  project                  = var.project_id
}

resource "google_compute_router" "router" {
  name    = "roce-router"
  project = var.project_id
  region  = var.region
  network = google_compute_network.vpc.id
}

resource "google_compute_router_nat" "nat" {
  name                               = "roce-nat"
  project                            = var.project_id
  router                             = google_compute_router.router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}

resource "google_compute_firewall" "allow_internal_serving_8000" {
  name    = "allow-internal-serving-8000"
  project = var.project_id
  network = google_compute_network.vpc.id

  allow {
    protocol = "tcp"
    ports    = ["8000"]
  }

  source_ranges = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
}

resource "google_compute_firewall" "allow_internal_primary_vpc" {
  name    = "allow-internal-primary-vpc"
  project = var.project_id
  network = google_compute_network.vpc.id

  allow {
    protocol = "tcp"
  }
  allow {
    protocol = "udp"
  }
  allow {
    protocol = "icmp"
  }

  source_ranges = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
}

resource "google_compute_firewall" "allow_ssh_roce_primary" {
  name    = "allow-ssh-roce-primary"
  project = var.project_id
  network = google_compute_network.vpc.id

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"]
}

resource "google_compute_network" "roce_network" {
  provider                = google-beta
  name                    = "rdma-net"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
  mtu                     = 8896
  network_profile         = "https://www.googleapis.com/compute/beta/projects/${var.project_id}/global/networkProfiles/${var.zone}-vpc-roce"
  project                 = var.project_id
}

resource "google_compute_subnetwork" "roce_subnet" {
  provider      = google-beta
  count         = 8
  name          = "rdma-sub-${count.index}"
  ip_cidr_range = cidrsubnet("10.200.0.0/16", 8, count.index)
  region        = var.region
  network       = google_compute_network.roce_network.id
  project       = var.project_id
}

resource "google_compute_firewall" "allow_roce_internal_rdma" {
  provider = google-beta
  name     = "allow-roce-internal-rdma"
  project  = var.project_id
  network  = google_compute_network.roce_network.id

  allow {
    protocol = "all"
  }

  source_ranges = ["10.200.0.0/16"]
}

resource "google_compute_network" "gvnic_network" {
  name                    = "gvnic-net"
  auto_create_subnetworks = false
  mtu                     = 8896
  project                 = var.project_id
}

resource "google_compute_subnetwork" "gvnic_subnet" {
  name                     = "gvnic-sub"
  ip_cidr_range            = "192.168.32.0/24"
  region                   = var.region
  network                  = google_compute_network.gvnic_network.id
  private_ip_google_access = true
  project                  = var.project_id
}

resource "google_compute_firewall" "allow_internal_gvnic" {
  name    = "allow-internal-gvnic"
  project = var.project_id
  network = google_compute_network.gvnic_network.name
  allow {
    protocol = "tcp"
  }
  allow {
    protocol = "udp"
  }
  allow {
    protocol = "icmp"
  }
  source_ranges = ["192.168.32.0/24"]
}
