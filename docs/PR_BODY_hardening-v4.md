# PR Summary: Pre-deployment Hardening (`hardening-v4`)

## Task Summary Table

| Task ID | Commit SHA | Outcome & Summary |
| :--- | :--- | :--- |
| **F6-1** | `5016594` | Install `fla-core` and `hf_transfer` in `Dockerfile.sglang` |
| **F6-2** | `40d3b1b` | Set `PYTORCH_CUDA_ALLOC_CONF expandable_segments:True` across workload templates |
| **F6-3** | `483cdba` | Add `--served-model-name moonshotai/Kimi-K3` to SGLang server and align model IDs |
| **F6-4** | `75849ff` | Wire `--mamba-full-memory-ratio 7.21` in SGLang `launch_server` exec line |
| **F6-5** | `96b7b9b` | Parameterize `FABRIC_GATE_TIMEOUT_SECONDS` and raise default to 900s |
| **F6-6** | `76e1139` | Install `cleanup_fabric_pods` on fabric gate abort paths |
| **F6-7** | `5899294` | Call `check_workflow_content` in `check_floors` to eliminate dead code |
| **F6-8** | `432a1de` | Correct factual defects in `PHASE6_RUNBOOK.md` |
| **F6-9** | `bb2fab1` | Delete busbw fallback column parser and update nccl-tests fixture |
| **F6-10** | `c7c23ac` | Record decision against `--dcp-size` in SGLang template |
| **F6-11** | `80a3f0b` | Correct KV capacity arithmetic and scaling table in `README.md` |
| **F6-12** | `2bb5310` | Tighten primary-VPC firewall `source_ranges` and update test assertions |
| **F6-13** | `3eed471` | Enforce `LIVE_VALIDATION=yes` guard in `02_deploy_infra.sh` and `03_deploy_workloads.sh` |
| **F6-14** | `3eed471` | Full S1–S9 gauntlet re-run clean (independent verification) |
| **P7-1** | `29f0a73` | Removed `weights_cache` bucket from Terraform, renamed purge variable to `PURGE_WEIGHTS_BACKUP`, fixed `/mxfp4` GCS path bug, and updated runbook/tests |
| **P7-2** | `f3f708c` | Shipped dormant `SGLANG_PARALLEL_PROFILE` (`tp16` / `tp8pp2`), verified byte-identical default render, and documented layer split caveat |
| **P7-3** | `a30f86c` | Completed runbook abort/rollback actions for Steps 4–8 with `file:line` citations, cost estimates, and procedural Step 6 gate |
| **P7-4** | `c03abdf` | Added fail-then-pass test case and script pipeline assertion to `test_nccl_bandwidth_parse.py`, closed DAY-0 comments, renamed README title, and authored PR body |

---

## Known Unmeasured Claims (`TBD — to be measured at first deployment`)

The following operational metrics and performance claims depend on live cluster hardware and are marked as unmeasured until first deployment:

1. **`docs/PHASE6_RUNBOOK.md:61`**: Procedural Step 6 gate baseline figures (TTFT, TPOT, and throughput targets `TBD — to be measured at first deployment`).
2. **`README.md:44`**: Token-Bucket Rate Limiting (TPM / RPM) & Exact-Match Prompt Caching latency on Cloud Memorystore Redis (`TBD — to be measured at first deployment`).
3. **`README.md:70`**: Instant Pod Hydration warm recovery duration (`TBD — to be measured at first deployment`).
4. **`README.md:81`**: Cloud Memorystore for Redis verified in-VPC exact-match prompt caching latency (`TBD — to be measured at first deployment`).
5. **`README.md:84`**: High-speed weight hydration backup bucket duration and throughput (`TBD — to be measured at first deployment`).
6. **`README.md:208`**: Optimal `SGLANG_PP_LAYER_PARTITION` layer partition for `tp8pp2` fallback profile across odd 93-layer architecture (`TBD — to be measured at first deployment`).
7. **`README.md:282`**: High-speed GCS weight backup hydration throughput (`TBD — to be measured at first deployment`).
8. **`README.md:286`**: Total duration to hydrate 1,453.7 GiB weights from GCS backup (`TBD — to be measured at first deployment`).
9. **`README.md:292`**: Transfer time for 1,453.7 GiB weight footprint into 2,000 GB Hyperdisk ML volume (`TBD — to be measured at first deployment`).
10. **`README.md:375`**: Warm Pod ROX Recovery duration (`TBD — to be measured at first deployment`).
11. **`README.md:377`**: Time for replacement 2-node serving replica to reach `Ready` state post-preemption (`TBD — to be measured at first deployment`).
12. **`README.md:404`**: Measured execution time of GCS weight hydration job (`TBD — to be measured at first deployment`).
13. **`README.md:500`**: Redis exact-match prompt caching verification in 5-point verification suite (`TBD — to be measured at first deployment`).
