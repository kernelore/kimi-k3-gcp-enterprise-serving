resource "google_bigquery_dataset" "enterprise_audit" {
  dataset_id                 = "kimi_k3_enterprise_audit"
  location                   = var.region
  project                    = var.project_id
  description                = "Enterprise audit logging dataset for Kimi K3 inference trajectories"
  delete_contents_on_destroy = true

  labels = {
    env   = var.env_label
    owner = var.owner_label
    ttl   = var.ttl_label
  }
}

resource "google_bigquery_table" "trajectories" {
  dataset_id          = google_bigquery_dataset.enterprise_audit.dataset_id
  table_id            = "trajectories"
  project             = var.project_id
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "request_timestamp"
  }

  schema = jsonencode([
    {
      name        = "request_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Unique ID of the request"
    },
    {
      name        = "request_timestamp"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "Timestamp when the request was received"
    },
    {
      name        = "virtual_key"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Virtual API key used for authentication"
    },
    {
      name        = "team_id"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Team identifier associated with the virtual key"
    },
    {
      name        = "model"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Model requested or served"
    },
    {
      name        = "prompt_tokens"
      type        = "INTEGER"
      mode        = "NULLABLE"
      description = "Number of prompt tokens processed"
    },
    {
      name        = "completion_tokens"
      type        = "INTEGER"
      mode        = "NULLABLE"
      description = "Number of completion tokens generated"
    },
    {
      name        = "total_cost_usd"
      type        = "FLOAT"
      mode        = "NULLABLE"
      description = "Total cost of the query in USD"
    },
    {
      name        = "ttft_ms"
      type        = "FLOAT"
      mode        = "NULLABLE"
      description = "Time to first token in milliseconds"
    },
    {
      name        = "tpot_ms"
      type        = "FLOAT"
      mode        = "NULLABLE"
      description = "Time per output token in milliseconds"
    }
  ])

  labels = {
    env   = var.env_label
    owner = var.owner_label
    ttl   = var.ttl_label
  }
}
