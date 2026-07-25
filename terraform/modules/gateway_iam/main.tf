resource "google_service_account" "gateway_sa" {
  account_id   = "kimi-k3-gateway-sa"
  display_name = "Kimi K3 Sovereign Gateway Service Account"
  project      = var.project_id
}

resource "google_project_iam_member" "cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.gateway_sa.email}"
}

resource "google_bigquery_dataset_iam_member" "bigquery_data_editor" {
  project    = var.project_id
  dataset_id = var.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = "serviceAccount:${google_service_account.gateway_sa.email}"
}

resource "google_service_account_iam_member" "workload_identity_user" {
  service_account_id = google_service_account.gateway_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[llm-serving/kimi-k3-gateway-sa]"
}
