#!/usr/bin/env python3
# ==============================================================================
# check_bq.py - BigQuery Audit Sink & Trajectory Verification (Kimi K3)
# ==============================================================================
# Verifies that enterprise AI audit logs and telemetry trajectories are being
# correctly streamed into BigQuery. Implements 3-tier fallback transport:
# 1. Standard 'bq' CLI query
# 2. Direct BigQuery REST API via Compute Engine instance metadata token
# 3. Google Cloud BigQuery Python Client library
# ==============================================================================

import json
import os
import subprocess
import sys
import urllib.request

# Normalize and sanitize proxy settings to prevent SSL/hostname transport errors
for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    val = os.environ.get(proxy_var, "")
    if val and not val.startswith("http://") and not val.startswith("https://"):
        os.environ.pop(proxy_var, None)

project_id = os.environ.get("PROJECT_ID", "YOUR_PROJECT_ID")
dataset_id = os.environ.get("AUDIT_DATASET_ID", "kimi_k3_enterprise_audit")
table_id = os.environ.get("AUDIT_TABLE_ID", "trajectories")

if project_id == "YOUR_PROJECT_ID" or not project_id:
    print("[FAIL] Error: Please set the PROJECT_ID environment variable (e.g. export PROJECT_ID=my-project-id)")
    sys.exit(1)

table_ref = f"{project_id}.{dataset_id}.{table_id}"
print(f"--> Verifying BigQuery Audit Sink on table: `{table_ref}`...")

success = False
total_rows = 0
sample_row = None

# Attempt 1: Standard 'bq' CLI query (primary path, resilient to SSL proxy quirks)
try:
    cmd = [
        "bq", "query",
        "--nouse_legacy_sql",
        "--format=json",
        f"SELECT count(*) as total_trajectories FROM `{table_ref}`"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if res.returncode == 0:
        data = json.loads(res.stdout)
        if data and isinstance(data, list):
            total_rows = int(data[0].get("total_trajectories", 0))
        
        if total_rows > 0:
            try:
                cmd_sample = [
                    "bq", "query",
                    "--nouse_legacy_sql",
                    "--format=json",
                    f"SELECT request_id, request_timestamp, model, prompt_tokens, completion_tokens, ttft_ms, tpot_ms FROM `{table_ref}` ORDER BY request_timestamp DESC LIMIT 1"
                ]
                res_sample = subprocess.run(cmd_sample, capture_output=True, text=True, timeout=30)
                if res_sample.returncode == 0:
                    sample_data = json.loads(res_sample.stdout)
                    if sample_data and isinstance(sample_data, list):
                        sample_row = sample_data[0]
            except Exception:
                cmd_sample_fb = ["bq", "query", "--nouse_legacy_sql", "--format=json", f"SELECT * FROM `{table_ref}` LIMIT 1"]
                res_fb = subprocess.run(cmd_sample_fb, capture_output=True, text=True, timeout=30)
                if res_fb.returncode == 0:
                    sample_row = json.loads(res_fb.stdout)[0] if json.loads(res_fb.stdout) else None
        success = True
    else:
        print(f"    [NOTE] bq CLI query stderr: {res.stderr.strip()}")
except Exception as bq_err:
    print(f"    [NOTE] bq CLI query failed: {bq_err}")

# Attempt 2: Direct BigQuery REST API fallback using instance metadata token
if not success:
    try:
        token_req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"}
        )
        with urllib.request.urlopen(token_req, timeout=3) as token_resp:
            token = json.loads(token_resp.read().decode())["access_token"]
            
        api_url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{project_id}/queries"
        req_body = json.dumps({"query": f"SELECT count(*) as total_trajectories FROM `{table_ref}`", "useLegacySql": False}).encode("utf-8")
        api_req = urllib.request.Request(api_url, data=req_body, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(api_req, timeout=10) as resp:
            api_data = json.loads(resp.read().decode())
            rows = api_data.get("rows", [])
            if rows:
                total_rows = int(rows[0]["f"][0]["v"])
            success = True
    except Exception as rest_err:
        print(f"    [NOTE] REST API fallback failed: {rest_err}")

# Attempt 3: Google Cloud BigQuery Python Client fallback
if not success:
    try:
        from google.cloud import bigquery
        from google.api_core.client_options import ClientOptions

        client_options = ClientOptions()
        client = bigquery.Client(project=project_id, client_options=client_options)

        query = f"SELECT count(*) as total_trajectories FROM `{table_ref}`"
        query_job = client.query(query)
        results = list(query_job.result())
        if results:
            total_rows = results[0].total_trajectories

        if total_rows > 0:
            try:
                query_sample = f"SELECT request_id, request_timestamp, model, prompt_tokens, completion_tokens, ttft_ms, tpot_ms FROM `{table_ref}` ORDER BY request_timestamp DESC LIMIT 1"
                sample_job = client.query(query_sample)
                sample_results = list(sample_job.result())
                if sample_results:
                    sample_row = dict(sample_results[0])
            except Exception as sample_err:
                print(f"    [NOTE] Could not sample row ({sample_err}).")
        success = True
    except Exception as py_err:
        print(f"    [NOTE] Python BigQuery client raised exception ({py_err}).")

if success:
    print(f"    [PASS] BigQuery audit verification succeeded! Total recorded trajectories: {total_rows}")
    if sample_row:
        print(f"    Sample Row Telemetry (Kimi K3 Schema): {sample_row}")
    sys.exit(0)
else:
    print(f"    [FAIL] BigQuery audit verification failed across all client, CLI, and REST transports.")
    sys.exit(1)
