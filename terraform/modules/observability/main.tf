resource "google_monitoring_dashboard" "kimi_k3_production_dashboard" {
  project        = var.project_id
  dashboard_json = jsonencode(yamldecode(file("${path.module}/dashboards/kimi_k3_monitoring_dashboard.yaml")))
}
