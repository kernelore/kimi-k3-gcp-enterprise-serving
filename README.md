# Kimi K3 Enterprise Serving Architecture

[![Google Cloud](https://img.shields.io/badge/Google_Cloud-Blackwell_B200-4285F4?style=flat-square&logo=googlecloud&logoColor=white)](https://cloud.google.com/compute/docs/gpus)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-MXFP4_MoE-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/)
[![TensorRT-LLM](https://img.shields.io/badge/Inference-TensorRT__LLM_Experimental-8A2BE2?style=flat-square)](https://github.com/NVIDIA/TensorRT-LLM)
[![SGLang](https://img.shields.io/badge/Inference-SGLang_Kimi__K3-8A2BE2?style=flat-square)](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](LICENSE)

> [!NOTE]
> **Disclaimer:** This repository is a personal engineering project and reference architecture. It is not an official Google product, is not covered by any Google Cloud Service Level Agreements (SLAs), and is not subject to official Google support channels. All code, scripts, and architectural models are provided "as-is" without warranty for educational, experimental, and benchmarking purposes.

**Target Hardware:** Google Kubernetes Engine (GKE) Blackwell (`2x
a4-highgpu-8g` nodes — 16x NVIDIA B200 HGX total per serving pod replica over
RoCEv2 fabric) \
**Target Workload:** High-Throughput Enterprise AI Engineering & Autonomous Agentic Workflows (`32k` to `128k` active context window out of `1,048,576` maximum
capacity) \
**Deployment Scope (Zonal Baseline → Regional HA Ready):** By default, the entire stack is provisioned with Zonal scope (`europe-north1-b` for the GKE cluster, Cloud SQL instance, Memorystore Redis cache, and Hyperdisk ML volume) to eliminate cross-zone network egress charges and accelerate deployment. The deployment can be easily extended to a Regional HA architecture by configuring `location = var.region` on the GKE cluster, setting `availability_type = "REGIONAL"` on Cloud SQL, upgrading Memorystore Redis to `tier = "STANDARD_HA"`, and distributing GPU worker replicas (`DP=N`) across multiple availability zones behind the internal load balancer.

---

## ⚡ Executive Summary & Full Tier 0-1-2 Architecture Overview

Deploying **Kimi K3** (`moonshotai/Kimi-K3`) at production scale represents a frontier engineering challenge. Featuring **2.8 Trillion total parameters** (**104 Billion activated per token**), **Stable LatentMoE** (`896` total routed experts, `16` active + `2` shared experts per token, 93 total layers: 69 KDA + 24 Gated MLA attention + 1 dense MLP layer), **Kimi Delta Attention (`KDA`)**, and **Attention Residuals (`AttnRes`)**, the model is trained end-to-end with **Quantization-Aware Training (QAT)** utilizing **`MXFP4` (Microscaling 4-bit weights)** and **`MXFP8` activations** via `compressed-tensors` (group size 32; attention, shared experts, dense MLP, lm_head, and vision encoder kept in bf16). Under `KimiK3ForConditionalGeneration`, the architecture is multimodal (KimiLinear text backbone + MoonViT-V2 401M vision encoder); this serving stack natively validates text serving.

Because the full model weight footprint in `MXFP4` occupies **1,453.7 GiB (96 safetensors shards)** on
disk and requires **`~2.8 TB` aggregate serving VRAM** (V<sub>weights</sub> +
V<sub>activations</sub> + V<sub>KV</sub> + V<sub>runtime</sub>), a single 8-GPU
node cannot serve Kimi K3. This architecture implements a **Multi-Node
Distributed Serving Replica (`2x a4-highgpu-8g` Blackwell nodes = `16x B200 HGX
GPUs` = `2,880 GB HBM3e` — 180 GB/GPU, 1,440 GB/node)** connected via Google Cloud GPUDirect RDMA over
RoCEv2 (`3.2 Tbps` per node inter-node interconnect, MTU 8896) operating under
**SGLang multi-node RoCEv2 serving (Primary Default)** per the official [SGLang Kimi-K3 Cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) with `--reasoning-parser kimi_k3 --tool-call-parser kimi_k3`. An experimental **NVIDIA TensorRT-LLM MPI** configuration is maintained in the codebase, though no published TensorRT-LLM K3 support exists as of launch.

```
+-----------------------------------------------------------------------------------------------------------------+
|                                       Private VPC High-Performance Network                                      |
|                 (Private Nodes, IAM-Gated Control Plane / Optional Private Endpoint, Private ILB)                |
+--------------------------------------------------------+--------------------------------------------------------+
                                                         |
                                                         v
+-----------------------------------------------------------------------------------------------------------------+
|                      Tier 1: Enterprise AI Gateway Layer (LiteLLM + Cloud SQL + Redis)                          |
|  - Virtual API Key Authentication (Internal Load Balancer Port 4000)                                            |
|  - Token-Bucket Rate Limiting (TPM / RPM) & Exact-Match Prompt Caching on Cloud Memorystore Redis (*TBD — to be measured at first deployment*)      |
|  - Asynchronous Telemetry & Trajectory Audit Sink streaming to BigQuery (`kimi_k3_enterprise_audit`)            |
|  - Upstream distribution across N active serving replicas via Kubernetes ClusterIP Service                      |
+--------------------------------------------------------+--------------------------------------------------------+
                                                         |
                                                         v
+-----------------------------------------------------------------------------------------------------------------+
|          GKE Blackwell Auto-Scaling Node Pool (`a4-highgpu-8g` | min: 0, max: `gpu_pool_max_nodes` (default 2))    |
|               Distributed Selectable Dual-Engine (TensorRT-LLM vs SGLang) Leader-Worker StatefulSet             |
|                                                                                                                 |
|  +--------------------------------------------------+         +----------------------------------------------+  |
|  |             Leader Node (Node 1 - 8x B200)       |<=======>|        Worker Node (Node 2 - 8x B200)        |  |
|  |  - Intra-Node: NVLink (1.8 TB/s) TP=8            | RoCEv2  |  - Intra-Node: NVLink (1.8 TB/s) TP=8        |  |
|  |  - Inter-Node: GPUDirect RDMA PP=1..2 / EP=8..16 | 3.2Tbps |  - Inter-Node: GPUDirect RDMA PP=1..2 / EP=8..16 |  |
|  |  - SGLang / TRT-LLM Execution Engine             | MTU 8896|  - SGLang / TRT-LLM MoE MXFP4 Kernels        |  |
|  |  - Dedicated KV Cache Pool: ~525 GB HBM3e        |         |  - Dedicated KV Cache Pool: ~525 GB HBM3e    |  |
|  +--------------------------+-----------------------+         +----------------------+-----------------------+  |
+-----------------------------|--------------------------------------------------------|--------------------------+
                              |                                                        |
                              +---------------------------+----------------------------+
                                                          | (Concurrent ReadOnlyMany Attach)
                                                          v
+-----------------------------------------------------------------------------------------------------------------+
|                                   Tier 0: Hyperdisk ML (`ROX` Multi-Node Storage)                               |
|                         `2,000 GB` (`2 TB`) Shared Model Weight Storage                                         |
|  - Shared ReadOnlyMany (`ROX`) persistent storage eliminating node cold-start downloads                         |
|  - Instant Pod Hydration (*TBD — to be measured at first deployment* warm recovery) concurrently attached to all N serving nodes |
+-----------------------------------------------------------------------------------------------------------------+
```

### ☁️ Google Cloud Products & Architectural Roles

| Google Cloud Product | Resource Identifier in Stack | Architectural Role & Implementation Details |
| :--- | :--- | :--- |
| **Google Kubernetes Engine (GKE)** | `module.cluster` | Private nodes with public, IAM-gated control-plane endpoint (or fully private with `enable_private_endpoint = true`) orchestrating dual-engine (SGLang / TRT-LLM) pods, Workload Identity Federation, and node pool autoscaling. |
| **Compute Engine A4 VMs** | `module.node_pool_spot` | `a4-highgpu-8g` Blackwell instances equipped with 8x NVIDIA B200 GPUs (1,440 GB HBM3e, 180 GB/GPU) and 32x local NVMe SSDs (12 TiB) per node, operating in 2-node pairs over RoCEv2. |
| **Hyperdisk ML (`ROX`)** | `module.storage` | 2,000 GB (2 TB) high-throughput block volume flipped to `ReadOnlyMany` mode for shared, zero-cold-start model weight mounting. |
| **Cloud Memorystore for Redis** | `module.cache` | In-memory tier for exact-match prompt caching (single-digit-ms in-VPC, *TBD — to be measured at first deployment* verified via port-forward) and gateway token-bucket rate limiting (RPM/TPM). |
| **Cloud SQL for PostgreSQL** | `module.database` | Private database accessed via Cloud SQL Auth Proxy storing virtual API keys, user budgets, and gateway routing configurations. |
| **BigQuery** | `module.audit` | Serverless audit dataset (`kimi_k3_enterprise_audit`) for asynchronous logging of conversation trajectories and token telemetry. |
| **Cloud Storage (GCS)** | `TF_STATE_BUCKET` / `GCS_WEIGHTS_BUCKET` | Remote Terraform state versioning and high-speed weight hydration backup bucket (hydration duration and throughput: *TBD — to be measured at first deployment*). |
| **Artifact Registry** | `module.storage` | Secure private container registry hosting pinned custom SGLang (`lmsysorg/sglang:kimi-k3@sha256:81a9c00654b3e4c7c681a4728a64fcb4853aa698dc9fea1959bbf4eb26bfb2e5`) / experimental TensorRT-LLM Blackwell serving images. |
| **Cloud Build** | `scripts/03_deploy_workloads.sh` | Serverless build pipeline for automated, self-healing image compilation from `docker/Dockerfile`. |
| **Virtual Private Cloud (VPC)** | `module.network` | Private network topology with Private Services Access (PSA), IAP SSH restrictions, and secondary RoCEv2 fabric (MTU 8896). |
| **Managed Service for Prometheus (GMP)** | `module.observability` | Native metrics pipeline capturing NVIDIA DCGM GPU metrics and serving request telemetry. |

---

## 📐 Mathematical Capacity Derivations & Memory Footprint

Designing for Kimi K3 requires rigorous memory and disk capacity engineering to ensure zero out-of-memory (OOM) failures during high-concurrency 128k context inferencing.

### 1. Static Disk Capacity (C<sub>disk</sub>)

The static weight footprint for 2.8 Trillion parameters quantized to `MXFP4` with scaling factors is measured as:

$$C_{\text{weights+scales}} = 1,453.7\,\text{GiB } (96\,\text{safetensors shards})$$

For both SGLang (TP=16, PP=1, EP=16) and TensorRT-LLM (TP=8, PP=2, EP=8), the volume holds weights only; TRT-LLM PyTorch backend loads checkpoints directly. To accommodate the static MXFP4 safetensors shards, the persistent Hyperdisk ML claim is explicitly sized at **`2,000 GB` (`2 TB`)**.

### 2. Serving VRAM Footprint (V<sub>total</sub>)

Total serving VRAM across a distributed replica pool must satisfy static model weights, dynamic activation buffers in `MXFP8`, PagedAttention KV cache, and CUDA/TensorRT runtime overhead:

$$V_{\text{total}} = V_{\text{weights (MXFP4)}} + V_{\text{activations (MXFP8)}} + V_{\text{CUDA/TRT-LLM runtime}} + V_{\text{KV-cache pool}}$$

$$V_{\text{total}} \approx 1,400\,\text{GB} + 150\,\text{GB} + 200\,\text{GB} + 1,130\,\text{GB} = 2,880\,\text{GB HBM3e}$$

Across a **2-Node Blackwell Replica Pool (`16x B200 HGX GPUs`)**, aggregate high-bandwidth memory is:

$$V_{\text{pool}} = 2\,\text{nodes} \times 8\,\text{GPUs/node} \times 180\,\text{GB/GPU} = 2,880\,\text{GB HBM3e (1,440 GB/node)}$$

Subtracting static weights (1,400 GB), activation buffers (150 GB), and CUDA/TRT-LLM runtime overhead (200 GB) leaves **1,130 GB of dedicated HBM3e memory for PagedAttention KV Cache** across the 16 GPUs (~70.6 GB KV cache per GPU).

### 3. Concurrent 128k Context Session Capacity

For an active context window of 128,000 tokens (128k), assuming FP8 KV cache quantization with Kimi Delta Attention compression:

$$\text{KV Footprint per 128k Session} \approx 26.9\,\text{GB}$$

$$\text{Max Concurrent 128k Sessions per Replica} = \left\lfloor \frac{1,130\,\text{GB}}{26.9\,\text{GB}} \right\rfloor = 42\text{ concurrent streams}$$
*Estimate, pending release.*

$$\text{Concurrent 128k Sessions per GPU} \approx 2.6\text{ streams/GPU}$$
*Estimate, pending release.*

### 4. Cluster Capacity Scaling Reference Table (DP=N Replicas)

While **1x 2-Node Replica (DP=1, 16x B200 GPUs) serves as the turnkey MVP baseline**, the architecture scales out horizontally to N replicas via Data Parallelism (DP=N) over ReadOnlyMany Hyperdisk ML:

| Cluster Scale (DP=N Replicas) | Physical Nodes (`a4-highgpu-8g`) | Total B200 GPUs | Total HBM3e Memory | Dedicated KV Cache Pool | Max Concurrent 128k Streams (*estimate, pending release*) | Aggregate Output Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1x Replica (DP=1, MVP Baseline)** | 2 Nodes | 16 | 2,880 GB (180 GB/GPU) | 1,130 GB | ~42 sessions | 1.0x (R<sub>base</sub>) |
| **2x Replicas (DP=2)** | 4 Nodes | 32 | 5,760 GB | 2,260 GB | ~84 sessions | 2.0x (2 R<sub>base</sub>) |
| **4x Replicas (DP=4)** | 8 Nodes | 64 | 11,520 GB | 4,520 GB | ~168 sessions | 4.0x (4 R<sub>base</sub>) |
| **Nx Replicas (DP=N)** | 2N Nodes | 16N | N x 2,880 GB | N x 1,130 GB | N x 42 sessions | Nx (N R<sub>base</sub>) |

> **Per-GPU Normalization Rule:** In all benchmark normalization calculations and performance reporting, aggregate cluster throughput must be divided by **`16.0`** (for a 1-replica 2-node pool) rather than 8.0, reflecting the 16x B200 GPUs required to host Kimi K3's 2.8T parameter footprint.

> [!NOTE]
> **Horizontal Autoscaling Absence:** Unlike reference stacks such as GLM, Kimi K3 serves as a single 2-node MPI replica across 16 B200s (`SERVING_REPLICAS=1`, `NODES_PER_REPLICA=2`, and `GPU_MAX_NODES=2`), leaving no spare GPU capacity in the cluster to scale into. Furthermore, dynamic pod scale-out events would require re-hydrating 1,453.7 GiB of model weights from GCS or local disk. Horizontal Pod Autoscaling (HPA) is therefore intentionally not implemented in this repository.

---

## 🔬 MLOps Considerations, Gateway Intelligence & CoT Guardrails

### 1. MoE All-to-All Expert Routing over RoCEv2

Kimi K3's Stable LatentMoE architecture routes tokens across **896 total experts** with **16 active experts per token**. In a 2-node TensorRT-LLM MPI deployment (`--tp_size 8 --pp_size 2 --ep_size 8`), expert parallelism (EP=8) requires continuous all-to-all tensor dispatch across physical node boundaries. This architecture enables secondary multi-NIC RoCEv2 interfaces (MTU 8896) over Google Cloud Titanium adapters, providing **`3.2 Tbps` dedicated RDMA bandwidth per node** to prevent networking bottlenecks during expert routing.

### 2. Gateway Chain-of-Thought (CoT) Preservation

Kimi K3 generates structured reasoning trajectories enclosed within `<think>...</think>` tags prior to emitting final responses. To preserve reasoning integrity in enterprise applications, the Tier 1 LiteLLM Gateway enforces strict configuration policies:

```yaml
general_settings:
  drop_params: false
```

Truncating or stripping thought blocks during multi-turn agentic execution causes catastrophic reasoning degradation and context misalignment. The gateway guarantees end-to-end transmission of raw reasoning tokens to client applications and BigQuery audit logs.

### 3. Clarification Guardrails against Action Overfitting (Roadmap / Unimplemented)

Large-scale agentic models can exhibit "action overfitting"—prematurely invoking external tools or APIs when user prompts lack critical parameters. As a planned roadmap capability (currently unimplemented), the LiteLLM Gateway architecture evaluates an automated clarification guardrail pipeline to intercept tool-use requests with high semantic ambiguity, injecting a clarification prompt guardrail before allowing tool execution.

### 4. Selectable Inference Engine (TensorRT-LLM vs SGLang) & RoCEv2 RDMA Co-Design

To support diverse production serving requirements and experimental research on
NVIDIA Blackwell B200 HGX, this architecture introduces a **Selectable
Dual-Engine Serving Framework**. Operators can seamlessly switch between
**SGLang** (the Primary Default) and **NVIDIA TensorRT-LLM** (the Experimental
Option) via a single environment variable in `scripts/config.env`:

```bash
# Set to "sglang" (default) or "trtllm" (experimental option)
export INFERENCE_ENGINE="sglang"
```

#### Dual-Engine Architecture & Multi-Node Distributed Execution

Because Kimi K3's 2.8 Trillion total parameter footprint (1,453.7 GiB MXFP4
weights) exceeds single-node HBM3e capacity, both engines operate in a **2-Node
Distributed Replica (`16x NVIDIA B200 HGX GPUs`)** over Google Cloud's RoCEv2
RDMA fabric. However, their distributed coordination mechanisms differ
significantly:

1.  **SGLang (Primary Default | `INFERENCE_ENGINE="sglang"`)**:

    -   **Ray-Less Native Distributed Execution**: Implements multi-node
        distributed execution **WITHOUT INSTALLING OR RUNNING RAY**. It
        leverages native PyTorch distributed / SGLang native `--dist-init-addr
        <leader_ip>:port` with `--nnodes 2 --node-rank <rank> --tp 16 --pp 1 --ep
        16`. Pinned to `lmsysorg/sglang:kimi-k3` per the official [SGLang Kimi-K3 Cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3).
    -   **Native Parsers**: Configured with `--reasoning-parser kimi_k3 --tool-call-parser kimi_k3` as prescribed in the SGLang cookbook.
    -   **Inter-Node Interconnect**: SGLang relies on NCCL
        GPUDirect RDMA over RoCEv2 (tuned via GKE gIB `set_nccl_env.sh`) for high-speed inter-host tensor passing
        across the 16x B200 GPUs.
    -   **Performance Optimizations**: Utilizes `--moe-runner-backend flashinfer_mxfp4 --decode-attention-backend flashmla --kv-cache-dtype fp8_e4m3` with RadixAttention prefix caching.

2.  **NVIDIA TensorRT-LLM (Experimental Option | `INFERENCE_ENGINE="trtllm"`)**:

    -   **Status**: Maintained as an experimental secondary engine option in this repository. **No published TensorRT-LLM K3 support exists as of launch**.
    -   **Distributed Orchestration**: OpenMPI manages rank assignment across
        the 2+ nodes via headless DNS discovery (`mpirun -n 16
        --allow-run-as-root --hostfile /tmp/hostfile trtllm-llmapi-launch trtllm-serve /mnt/rox/Kimi-K3 --backend pytorch`).
    -   **Inter-Node Interconnect**: NCCL handles all inter-node B200 tensor and
        pipeline synchronization directly via GPUDirect RDMA over RoCEv2
        (tuned via GKE gIB `set_nccl_env.sh`).

#### Mandatory RoCEv2 GPUDirect RDMA & Blackwell Hardware Specifications

To prevent inter-node communication bottlenecks during MoE all-to-all expert
routing across 896 experts, both serving engines enforce mandatory RoCEv2
networking configurations:

-   **Jumbo Frame High MTU (`mtu = 8896`)**: Enforced by default across all 8
    secondary RoCEv2 subnets (`rdma-sub-0` through `rdma-sub-7`) under VPC
    `rdma-net` in `terraform/modules/network/`.
-   **Compact Zonal Placement Policy (`COLLOCATED`)**: Configured in
    `terraform/modules/node_pool_spot/` (`pp-kimi-b200-roce`) and attached to
    all GPU nodes to ensure sub-microsecond physical network switch hop latency.
-   **Shared Memory & IPC Capabilities**: Container runtime definitions grant
    `IPC_LOCK` capability and mount a dedicated 512 GiB memory-backed `/dev/shm`
    volume (`emptyDir: { medium: "Memory", sizeLimit: "512Gi" }`) for zero-copy
    intra-node IPC.

#### Engine Feature & Architecture Comparison Table

Feature / Metric              | SGLang (`sglang`)                                      | NVIDIA TensorRT-LLM (`trtllm`)
:---------------------------- | :----------------------------------------------------- | :----------------
**Status in Repository**      | **Primary Default** (`INFERENCE_ENGINE="sglang"`)      | **Experimental** (No published TRT-LLM K3 support as of launch)
**Container Image**           | `lmsysorg/sglang:kimi-k3`                              | Custom TRT-LLM builder
**Distributed Orchestration** | Native PyTorch Distributed (`--dist-init-addr`, **No Ray**) | OpenMPI (`mpirun -n 16 --hostfile /tmp/hostfile`)
**Inter-Node Networking**     | RoCEv2 GPUDirect RDMA (tuned via GKE gIB `set_nccl_env.sh`) | RoCEv2 GPUDirect RDMA (tuned via GKE gIB `set_nccl_env.sh`)
**Weight & Kernel Format**    | Native `KimiK3ForConditionalGeneration` / FlashInfer / FlashMLA | Direct PyTorch checkpoint loading (Experimental)
**Parsers**                   | `--reasoning-parser kimi_k3 --tool-call-parser kimi_k3` | Custom regex parser
**Prefix Caching**            | RadixAttention (Optimal for multi-turn & tree search)  | Standard KV Cache block reuse
**Memory Management**         | `--mem-fraction-static 0.90`                           | `--kv_cache_free_gpu_memory_fraction 0.90`
**Ideal Workload Profile**    | Dynamic interactive sessions, structured JSON, reasoning prompts | Maximum raw throughput batch serving (experimental)

#### Live Benchmark Performance Comparison

<!-- ENGINE_COMPARISON_START -->
| Metric / Engine Profile | SGLang (Primary Default) | NVIDIA TensorRT-LLM (Experimental) |
| :--- | :--- | :--- |
| **Status** | *pending first live benchmark run after the 2026-07-27 weight release* | *pending first live benchmark run after the 2026-07-27 weight release* |
| **Time-to-First-Token (TTFT)** | — | — |
| **Time-per-Output-Token (TPOT)** | — | — |
| **Normalized Per-GPU Throughput** | — | — |
<!-- ENGINE_COMPARISON_END -->

---

## 📦 Hybrid 3-Tier Storage Co-Design & Weight Cache Lifecycle

To eliminate multi-hour weight downloads when scaling compute pods (the volume holds weights only; TRT-LLM PyTorch backend loads checkpoints directly without pre-compiled engine files), the architecture implements a hybrid 3-tier storage co-design:

```
+---------------------------------------------------------------------------------------------------+
|                        Tier 0: Hyperdisk ML ReadOnlyMany (`ROX`) Volume                           |
|  - Capacity: 2,000 GB (2 TB) Block Storage | Formatted XFS (mountOptions: nouuid, ro, norecovery) |
|  - Content: Pre-staged MXFP4 safetensors shards (TRT-LLM PyTorch backend loads checkpoints directly) |
+---------------------------------------------------------------------------------------------------+
                                                  ^
                                                  | (Concurrent Read-Only Mount across 2N Nodes)
+---------------------------------------------------------------------------------------------------+
|                     Tier 2: Local NVMe SSD RAID 0 Scratch Array (`/mnt/scratch`)                 |
|  - Capacity: 32x 375 GiB Local NVMe SSDs = 12 TiB per node | Formatted XFS via RAID 0 DaemonSet   |
|  - Content: Ultra-fast local buffer for MPI shared memory exchange and runtime staging logs       |
+---------------------------------------------------------------------------------------------------+
                                                  ^
                                                  | (High-Speed Backup Hydration @ *TBD — to be measured at first deployment*)
+---------------------------------------------------------------------------------------------------+
|                   Tier 3: Cloud Storage (GCS) Backup Cache Bucket                                 |
|  - URI: `gs://<project>-kimi-k3-weights-backup/mxfp4` | Cost: ~$40/month (outside TF state)       |
|  - Content: Persistent backup of verified weight shards for rapid disaster recovery hydration (*TBD — to be measured at first deployment*) |
+---------------------------------------------------------------------------------------------------+
```

### Weight Cache Hydration Lifecycle

When deploying a fresh cluster or recovering from a disaster, downloading 1,453.7 GiB of weights from external model hubs can take hours. Using a pre-populated GCS weight cache bucket (`Tier 3`), the automated hydration job (`02-hydrate-weights-gcs.yaml.template`) transfers the entire 1,453.7 GiB weight footprint into the 2,000 GB Hyperdisk ML volume in ***TBD — to be measured at first deployment*** (hydration throughput scales with hyperdisk_ml_throughput_mibps (default 24,576 MiB/s; 6,144 MiB/s is a known-good fallback if regional quota is unavailable)).

#### Operational Seeding & Hydration Commands

```bash
# 1. Standard deploy leveraging an existing GCS weight cache bucket
export GCS_WEIGHTS_BUCKET="gs://YOUR_PROJECT_ID-kimi-k3-weights-cache/mxfp4"
./scripts/03_deploy_workloads.sh

# 2. Automatically populate / seed the GCS cache after an initial Hugging Face download
export POPULATE_WEIGHTS_CACHE="true"
export GCS_WEIGHTS_BUCKET="gs://YOUR_PROJECT_ID-kimi-k3-weights-cache/mxfp4"
./scripts/03_deploy_workloads.sh

# 3. Force re-staging of model weight shards (flips volume back to ReadWrite mode temporarily)
export FORCE_WEIGHT_JOB="true"
./scripts/03_deploy_workloads.sh
```

#### Complete Teardown & State Bucket Retention

By default, `./scripts/06_destroy_all.sh` and `terraform destroy` remove all Terraform-managed resources—including the GKE cluster, RoCEv2 network, Cloud SQL instance, Hyperdisk ML ReadOnlyMany (`ROX`) volume, and GCS weight cache bucket—to ensure zero ongoing cloud spend. For temporary overnight pauses that retain weights and disks, use the scheduled evening turndown CronJob or scale the serving StatefulSet to 0.

The only GCS bucket retained after `./scripts/06_destroy_all.sh` is the out-of-band Terraform remote state bucket (`gs://${PROJECT_ID}-kimi-k3-tfstate`), listed in an **OUT-OF-BAND RETAINED BUCKET INVENTORY** report. To explicitly delete the state bucket when no longer needed:

```bash
# Delete the out-of-band Terraform remote state bucket
gcloud storage rm --recursive "gs://${PROJECT_ID}-kimi-k3-tfstate"
```

--------------------------------------------------------------------------------

## 📁 Repository Directory Structure

```
kimi-k3-gcp-enterprise-serving/
├── LICENSE                        # Apache-2.0 License
├── README.md                      # Reference architecture and operational runbook (this file)
├── benchmarks/                    # Synthetic load testing and stress benchmark Python suites
│   ├── benchmark_kimi_k3.py       # Standard enterprise performance benchmark (TTFT, TPOT, Throughput)
│   ├── massive_benchmark_kimi_k3.py # High-concurrency stress test simulating 20 agent streams
│   ├── run_prefill_benchmark_kimi_k3.py # Empirical prompt-ingestion prefill benchmark (8k-in/16-out)
│   ├── run_saturation_sweep_kimi_k3.py # Direct engine saturation sweep across concurrency levels c=1..64
│   └── soak_benchmark_kimi_k3.py  # 30-minute continuous stability endurance test (1,800 seconds)
├── docker/                        # Container definitions for custom serving runtimes
│   ├── Dockerfile                 # Experimental TensorRT-LLM serving container image definition
│   └── Dockerfile.sglang          # Primary default SGLang serving container image (lmsysorg/sglang:kimi-k3@sha256:81a9c006...)
├── scripts/                       # Automated lifecycle Bash & Python scripts
│   ├── 01_setup_and_check.sh      # Preflight CLI checks, password generation, API enablement, tfvars sync
│   ├── 02_deploy_infra.sh         # Terraform infrastructure provisioning (VPC, GKE, Spot VMs, Cloud SQL, Redis)
│   ├── 03_deploy_workloads.sh     # Manifest rendering, NVMe RAID formatter, WIF RBAC, and gateway deployment
│   ├── 04_verify_cluster.sh       # 5-point verification suite (Nodes, Gateway, Virtual Keys, Caching, BigQuery)
│   ├── 05_run_benchmarks.sh       # Automated benchmark runner with in-cluster soak and workstation modes
│   ├── 06_destroy_all.sh          # Self-healing teardown script with database role and peering guards
│   ├── check_bq.py                # Python audit client querying real-time BigQuery trajectory streams
│   ├── config.env.example         # Template configuration for environment variables and resource tags
│   ├── requirements.txt           # Python client dependencies (BigQuery and Cloud Storage)
│   └── test_live_gateway.py       # Turnkey live chat completion script for verifying gateway inference
└── terraform/                     # Modular, enterprise-grade Terraform infrastructure definitions
    ├── main.tf                    # Root composition module integrating cluster, network, and storage modules
    ├── variables.tf / outputs.tf  # Global input variables and cluster/storage/database outputs
    ├── terraform.tfvars.example   # Example variables template
    ├── modules/                   # Infrastructure modules: cluster, network, node_pool_spot, storage,
    │                              # database, cache, audit, gateway_iam, observability
    └── manifests/                 # Kubernetes YAML templates and generated runtime manifests
```

---

## 🛡️ Failover Mechanics, Dynamic Quotas & Self-Healing Resilience

### 1. K8s Readiness Ejection & LiteLLM Router Retries

The Tier 1 Enterprise AI Gateway and Kubernetes infrastructure utilize active health probing via readiness probes and LiteLLM router retries. If a serving pod experiences GPU ECC faults, NVLink lockups, or network drops, Kubernetes readiness probes fail, ejecting the unhealthy replica from the upstream ClusterIP routing pool while LiteLLM automatically retries failed requests against healthy replicas with zero client-facing error leakage.

### 2. Dynamic Spot Preemption Priorities (P0 > P1 > P2)

To achieve 55%+ compute cost savings, serving compute pools utilize Google Cloud Blackwell Spot VMs (`a4-highgpu-8g`). To prevent cluster-wide outages during spot preemption events, workloads are scheduled across a strict 3-tier preemption hierarchy:

* **Priority P0 (Non-Preemptible System Core):** The Enterprise AI Gateway (LiteLLM), Cloud SQL Auth Proxy, and Redis instance run exclusively on a dedicated, non-preemptible e2-standard system node pool (`np-system`). Control-plane routing and authentication never go offline.
* **Priority P1 (Graceful Serving Workloads):** Serving pods run on the B200 spot pool with Kubernetes `SIGTERM` handling and `preStop` drain hooks. When GCP issues a 30-second spot preemption notice, the pod traps `SIGTERM`, executes the `preStop` drain hook to deregister from the LiteLLM gateway, finishes inflight requests, and terminates cleanly.
* **Priority P2 (Batch Benchmarks Only):** Off-peak massive benchmark suites run at lowest priority, yielding resources instantly to P1 serving pods during quota contention.

### 3. Warm Pod ROX Recovery (*TBD — to be measured at first deployment*)

When a preempted B200 spot node is replaced by the GKE Cluster Autoscaler, traditional architectures suffer multi-hour cold starts downloading weights. Because Kimi K3's weights are mounted via ReadOnlyMany (`ROX`) Hyperdisk ML (the volume holds weights only; TRT-LLM PyTorch backend loads checkpoints directly without pre-compiled engine files), new pods bypass network downloads entirely. Once physical nodes pass CUDA initialization, the 2-node serving replica reaches `Ready` state in ***TBD — to be measured at first deployment***.

### 4. Self-Healing Teardown Loop & Database Guardrails (Issues 6 & 7 Guards)

Teardown automation (`06_destroy_all.sh`) incorporates proactive self-healing guardrails to prevent Terraform destruction deadlocks:

1. **Cloud SQL Role Dependency Guard (Issue 6):** Proactively drops the database `kimi_k3_gateway` via `gcloud sql databases delete` before running Terraform destroy. This releases PostgreSQL owned database roles, preventing Terraform from hanging on user deletion dependency violations.
2. **SDN Peering Propagation Guard (Issue 7):** If Google Cloud SDN propagation delays cause VPC peering deletion failures, the teardown script detects the failure, automatically executes `gcloud compute networks peerings delete` to sever dangling private service access peerings, and retries `terraform destroy` until clean completion.

---

## 🗺️ 4-Phase Implementation & Deployment Roadmap Summary

The deployment lifecycle is structured into four systematic, verifiable engineering phases:

```
+-----------------------------------------------------------------------------------------------------------------+
|                                 PHASE 1: Enterprise Infrastructure & Fabric Foundation                          |
|  - Provision Private VPC, Secondary RoCEv2 Fabric Subnets (MTU 8896), Cloud NAT, and IAP SSH Firewall Rules     |
|  - Deploy GKE Blackwell Cluster (Private Nodes, IAM-Gated Endpoint) and Workload Identity Federation (WIF)      |
|  - Provision Cloud SQL PostgreSQL Instance (`kimi-k3-gateway-db`) & Cloud Memorystore Redis (`kimi-k3-gateway-cache`)   |
+-----------------------------------------------------------------------------------------------------------------+
                                                         |
                                                         v
+-----------------------------------------------------------------------------------------------------------------+
|                                 PHASE 2: Hybrid Storage & Weight Hydration Pipeline                             |
|  - Provision 2,000 GB (2 TB) Hyperdisk ML Block Volume & Apply Local NVMe RAID 0 DaemonSet (`/mnt/scratch`)     |
|  - Execute GCS Weight Hydration Job (*TBD — to be measured at first deployment*) or Hugging Face Secure Downloader |
|  - Flip Hyperdisk ML Volume Access Mode to ReadOnlyMany (`ROX`) for Zero-Copy Multi-Node Scale-Out              |
+-----------------------------------------------------------------------------------------------------------------+
                                                         |
                                                         v
+-----------------------------------------------------------------------------------------------------------------+
|                              PHASE 3: Enterprise Gateway & Multi-Node Serving Deployment                        |
|  - Deploy Tier 1 LiteLLM Enterprise Gateway & Cloud SQL Proxy on Non-Preemptible System Pool (`np-system`)      |
|  - Deploy 2-Node SGLang / TRT-LLM Headless StatefulSet (Leader Node 0 + Worker Node 1 over RoCEv2 / NVLink)     |
|  - Configure Managed Service for Prometheus (GMP) Observability                                                 |
+-----------------------------------------------------------------------------------------------------------------+
                                                         |
                                                         v
+-----------------------------------------------------------------------------------------------------------------+
|                            PHASE 4: Production Verification & Benchmark Certification                           |
|  - Execute 5-Point Enterprise Gateway Verification Suite (Auth, Virtual Keys, Quotas, Redis Cache, BigQuery)    |
|  - Run In-Cluster Soak Benchmark (`05_run_benchmarks.sh --mode soak --in-cluster`) & Saturation Sweeps        |
|  - Certify Telemetry Audit Sink in BigQuery (`check_bq.py`) & Verify Zero Out-of-Memory (OOM) Execution         |
+-----------------------------------------------------------------------------------------------------------------+
```

---

## 🚀 Turnkey Quickstart & Operational Runbook

### Step 1: Virtual Environment Setup

To prevent PEP 668 system Python package conflicts on Debian/Ubuntu environments, configure and activate an isolated virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

### Step 2: Configure Environment & IAM Prerequisites

Copy the configuration template and customize parameters for your target environment:

```bash
cp scripts/config.env.example scripts/config.env
nano scripts/config.env
```

#### Mandatory IAM Operator Roles
Executing the deployment pipeline requires the following IAM roles bound to your Google Cloud operator identity:

* `roles/container.admin`: Required for provisioning GKE clusters and applying ClusterRoleBindings.
* `roles/servicenetworking.networksAdmin`: Required for establishing Private Services Access VPC peerings for Cloud SQL.
* `roles/iam.serviceAccountUser`: Required for attaching Workload Identity service accounts to Kubernetes pods.
* `roles/resourcemanager.projectIamAdmin` (or `roles/editor`): Required for binding WIF IAM policies.

```bash
ACCOUNT=$(gcloud config get-value account)
[[ "${ACCOUNT}" == *"gserviceaccount.com" ]] && MEMBER="serviceAccount:${ACCOUNT}" || MEMBER="user:${ACCOUNT}"

for ROLE in roles/container.admin \
            roles/servicenetworking.networksAdmin \
            roles/iam.serviceAccountUser \
            roles/resourcemanager.projectIamAdmin; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="${MEMBER}" --role="${ROLE}" --condition=None
done
```
`01_setup_and_check.sh` re-validates these roles and prints the exact missing grant command if any check fails.

### Step 3: Run Preflight Checks & Sync Configuration

Execute the setup orchestrator to validate CLI tools, check IAM permissions, verify Python library imports (`google.cloud.bigquery`, `google.cloud.storage`), generate cryptographic secrets (`DB_PASSWORD` and `GATEWAY_MASTER_KEY`), and populate `terraform/terraform.tfvars`:

```bash
./scripts/01_setup_and_check.sh
```

### Step 4: Provision Infrastructure via Terraform

Provision the VPC network, secondary RoCEv2 subnets, GKE cluster, Blackwell Spot
node pools, Cloud SQL instance, Cloud Memorystore Redis, BigQuery dataset, and
2,000 GB Hyperdisk ML volume:

```bash
./scripts/02_deploy_infra.sh
```
*Note: Includes automated state checking (Issue 10 Guard) to warn if existing remote Terraform state is detected.*

### Step 5: Render Manifests & Deploy Workloads

Render Kubernetes manifest templates, format local NVMe SSDs into RAID 0 scratch arrays, hydrate model weights, flip volume access mode to ReadOnlyMany (`ROX`), and deploy the Enterprise AI Gateway on port 4000:

```bash
./scripts/03_deploy_workloads.sh
```
*Note: Features self-healing runtime image verification—automatically triggering `gcloud builds submit` if the target container image is missing from Artifact Registry.*

### Step 6: Verify Cluster Health, Gateway Virtual Keys & BigQuery Audits

Execute the automated 5-point verification suite to certify node health, virtual API key generation, token-bucket rate limiting, Redis exact-match prompt caching (_TBD — to be measured at first deployment_), and BigQuery trajectory audit streaming:

```bash
./scripts/04_verify_cluster.sh
```

### Step 7: Execute Benchmark Suites

Evaluate Kimi K3 inference performance across Time-to-First-Token (`TTFT`), Time-per-Output-Token (`TPOT`), and aggregate throughput:

#### ⚡ In-Cluster Benchmark Execution (Recommended for Sustained & Soak Testing)
Workstation `kubectl port-forward` tunnels can experience socket drops under heavy load. For sustained stress testing or 30-minute endurance soaks, execute benchmark jobs directly inside the Kubernetes cluster on system nodes:

```bash
# Execute 30-minute endurance soak benchmark from inside GKE targeting internal Gateway service
./scripts/05_run_benchmarks.sh --mode soak --in-cluster
```

#### 🖥️ Workstation Smoke Testing (Quick Verification)
For immediate smoke testing from your local terminal via automated port-forwarding:

```bash
./scripts/05_run_benchmarks.sh --mode standard --target gateway
```
*Note: All per-GPU throughput metrics reported by benchmark harnesses are automatically normalized by dividing cluster throughput by **`16.0`**.*

#### 📊 Direct Engine Saturation Sweep & Prefill Ingestion Benchmarking

To run cache-bypassed direct GPU generation saturation sweeps across concurrency
levels ($c \in \{1, 8, 16, 32, 64\}$) or evaluate prompt prefill ingestion rates
directly on the 16x B200 HGX pool:

```bash
# 1. Run prompt prefill / ingestion benchmark (8,192 input tokens -> 16 output tokens)
python3 benchmarks/run_prefill_benchmark_kimi_k3.py

# 2. Run direct engine saturation sweep (max_tokens=256, ignore_eos=True, 0% cache hits)
python3 benchmarks/run_saturation_sweep_kimi_k3.py
```

---

## 🧹 Idempotent Teardown & Clean Purge

### Automated Teardown

When testing is complete, run the teardown script to safely drain Kubernetes
workloads, release Persistent Volumes, and run `terraform destroy`:

```bash
./scripts/06_destroy_all.sh
```

`06_destroy_all.sh` is completely idempotent and self-healing:

1.  Proactively deletes Cloud SQL databases (`kimi_k3_gateway`) to release owned
    database roles before Terraform user deletion (preventing role drop
    dependency errors).
2.  Sets `deletion_policy = "ABANDON"` on `google_service_networking_connection`
    and automatically cleans up dangling compute VPC peerings if Google Cloud
    SDN propagation delay occurs.
3.  Proactively cleans up the 2,000 GB Hyperdisk ML ReadOnlyMany (`ROX`) volume
    and local NVMe scratch arrays.
4.  Supports `PURGE_WEIGHTS_CACHE=true` for explicit pre-destroy weight cache bucket purge (the bucket is in any case destroyed by `terraform destroy` since `force_destroy = true`).

### Retained Storage & Bucket Purge Guide

`./scripts/06_destroy_all.sh` executes `terraform destroy`, which deletes all Terraform-managed resources including the GCS weight cache bucket and Hyperdisk ML volume. The only bucket remaining is the out-of-band Terraform remote state bucket, which is listed in an **OUT-OF-BAND RETAINED BUCKET INVENTORY** report. To completely remove the state bucket:

```bash
# 1. Purge and delete the Terraform remote state bucket
gcloud storage rm --recursive "gs://${PROJECT_ID}-kimi-k3-tfstate"

# 2. Purge and delete the Trajectory audit backup bucket
gcloud storage rm --recursive "gs://${PROJECT_ID}-kimi-k3-trajectories"

# 3. (Optional) Delete custom GCS weight cache bucket if created
gcloud storage rm --recursive "gs://${PROJECT_ID}-kimi-k3-weights-backup"
```
