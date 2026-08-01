# KIMI K3 Enterprise Inference Architecture

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
**Target Workload:** High-Throughput Enterprise AI Engineering & Autonomous Agentic Workflows (`32k` to `128k` active context window out of the `1,048,576` maximum
declared by the checkpoint's `max_position_embeddings`; this stack deploys `--context-length 131072` via `SGLANG_CONTEXT_LENGTH`) \
**Deployment Scope (Zonal Baseline → Regional HA Ready):** By default, the entire stack is provisioned with Zonal scope (`europe-north1-b` for the GKE cluster, Cloud SQL instance, Memorystore Redis cache, and Hyperdisk ML volume) to eliminate cross-zone network egress charges and accelerate deployment. The deployment can be easily extended to a Regional HA architecture by configuring `location = var.region` on the GKE cluster, setting `availability_type = "REGIONAL"` on Cloud SQL, upgrading Memorystore Redis to `tier = "STANDARD_HA"`, and distributing GPU worker replicas (`DP=N`) across multiple availability zones behind the internal load balancer.

---

## ⚡ Executive Summary & Full Tier 0-1-2 Architecture Overview

Deploying **Kimi K3** (`moonshotai/Kimi-K3`) at production scale represents a frontier engineering challenge. Featuring **2.8 Trillion total parameters** (**104 Billion activated per token**), **Stable LatentMoE** (`896` total routed experts, `16` active + `2` shared experts per token, 93 total layers = 69 KDA + 24 Gated MLA attention, with `first_k_dense_replace = 1` giving the first layer a dense MLP instead of a routed MoE block — it is not a 94th layer), **Kimi Delta Attention (`KDA`)**, and **Attention Residuals (`AttnRes`)**, the model is trained end-to-end with **Quantization-Aware Training (QAT)** utilizing **`MXFP4` (Microscaling 4-bit weights)** and **`MXFP8` activations** via `compressed-tensors` (group size 32; attention, shared experts, dense MLP, lm_head, and vision encoder kept in bf16). Under `KimiK3ForConditionalGeneration`, the architecture is multimodal (KimiLinear text backbone + MoonViT-V2 401M vision encoder); this serving stack natively validates text serving.

Because the full model weight footprint in `MXFP4` occupies **1,453.7 GiB (96 safetensors shards) (= 1,560.9 GB; plan ~1.56 TB)** on
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
|  |  - KV/State Pool (derived): ~443 GB              |         |  - KV/State Pool (derived): ~443 GB          |  |
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
|  - Instant Pod Hydration (measured 17 min 03 s warm recovery) concurrently attached to all N serving nodes      |
+-----------------------------------------------------------------------------------------------------------------+
```

### ☁️ Google Cloud Products & Architectural Roles

| Google Cloud Product | Resource Identifier in Stack | Architectural Role & Implementation Details |
| :--- | :--- | :--- |
| **Google Kubernetes Engine (GKE)** | `module.cluster` | Private nodes with public, IAM-gated control-plane endpoint (or fully private with `enable_private_endpoint = true`) orchestrating dual-engine (SGLang / TRT-LLM) pods, Workload Identity Federation, and node pool autoscaling. |
| **Compute Engine A4 VMs** | `module.node_pool_spot` | `a4-highgpu-8g` Blackwell instances equipped with 8x NVIDIA B200 GPUs (1,440 GB HBM3e, 180 GB/GPU) and 32x local NVMe SSDs (32 × 375 GiB = 12,000 GiB ≈ 11.7 TiB) per node, operating in 2-node pairs over RoCEv2. |
| **Hyperdisk ML (`ROX`)** | `module.storage` | 2,000 GB (2 TB) high-throughput block volume flipped to `ReadOnlyMany` mode for shared, zero-cold-start model weight mounting. |
| **Cloud Memorystore for Redis** | `module.cache` | In-memory tier for exact-match prompt caching (single-digit-ms in-VPC, *TBD — to be measured at first deployment* verified via port-forward) and gateway token-bucket rate limiting (RPM/TPM). Note: exact-match prompt caching applies at the gateway level; inside the serving engine, RadixAttention prefix caching reuse benefits only the 24 MLA attention layers, while KDA linear layers are not prefix-shareable. |
| **Cloud SQL for PostgreSQL** | `module.database` | Private database accessed via Cloud SQL Auth Proxy storing virtual API keys, user budgets, and gateway routing configurations. |
| **BigQuery** | `module.audit` | Serverless audit dataset (`kimi_k3_enterprise_audit`) for asynchronous logging of conversation trajectories and token telemetry. |
| **Cloud Storage (GCS)** | `TF_STATE_BUCKET` / `GCS_WEIGHTS_BUCKET` | Remote Terraform state versioning and high-speed weight hydration backup bucket (measured hydration: 1,453.7 GiB in 11 min 18 s, 2.1 GiB/s average). |
| **Artifact Registry** | `module.storage` | Secure private container registry hosting the custom SGLang and experimental TensorRT-LLM Blackwell serving images. The image actually built and deployed is `${REGION}-docker.pkg.dev/${PROJECT_ID}/kimi-prod/sglang-blackwell:latest` (`03_deploy_workloads.sh:237`) — a **mutable tag**, not a digest. The digest pin sits one level up, on the base image in `docker/Dockerfile.sglang`: `FROM lmsysorg/sglang:kimi-k3@sha256:81a9c00654b3e4c7c681a4728a64fcb4853aa698dc9fea1959bbf4eb26bfb2e5`. |
| **Cloud Build** | `scripts/03_deploy_workloads.sh` | Serverless build pipeline for automated, self-healing image compilation from `docker/Dockerfile`. |
| **Virtual Private Cloud (VPC)** | `module.network` | Private network topology with Private Services Access (PSA), IAP SSH restrictions, and secondary RoCEv2 fabric (MTU 8896). |
| **Managed Service for Prometheus (GMP)** | `module.observability` | Native metrics pipeline capturing NVIDIA DCGM GPU metrics and serving request telemetry. |

---

## 📐 Mathematical Capacity Derivations & Memory Footprint

Designing for Kimi K3 requires rigorous memory and disk capacity engineering to ensure zero out-of-memory (OOM) failures during high-concurrency 128k context inferencing.

### 1. Static Disk Capacity (C<sub>disk</sub>)

The static weight footprint for 2.8 Trillion parameters quantized to `MXFP4` with scaling factors is measured as:

$$C_{\text{weights+scales}} = 1,453.7\,\text{GiB } (96\,\text{safetensors shards})\,(= 1,560.9\,\text{GB}; \text{plan }\sim 1.56\,\text{TB})$$

For both SGLang (TP=16, PP=1, EP=16) and TensorRT-LLM (TP=8, PP=2, EP=8), the volume holds weights only; TRT-LLM PyTorch backend loads checkpoints directly. To accommodate the static MXFP4 safetensors shards, the persistent Hyperdisk ML claim is explicitly sized at **`2,000 GB` (`2 TB`)**.

### 2. Serving VRAM Footprint (V<sub>total</sub>)

Total serving VRAM across a distributed replica pool is derived step-by-step from static model weight allocation and memory fraction limits:

- **Total HBM Pool**: 16 × B200 × 180 GB = **2,880 GB** (1,440 GB per node)
- **Checkpoint Weight Sharding**: 1,560.94 GB checkpoint / 16 GPUs = **97.6 GB/GPU** of weights
- **Static Pool Sizing (`--mem-fraction-static 0.85`)**: 0.85 × 180 GB = **153 GB/GPU** static pool
- **KV/State Headroom per GPU**: 153 GB − 97.6 GB = **55.4 GB/GPU**
- **Aggregate KV/State Pool**: 55.4 GB × 16 GPUs ≈ **887 GB** (≈443 GB/node)

> [!IMPORTANT]
> Every figure in this section is a **paper derivation, not a measurement.** It also treats the KV pool as uniform, which Kimi K3 is not: only the 24 MLA layers consume per-token KV, while the 69 KDA layers hold a fixed-size recurrent state per *sequence*, sized by the engine's own default recurrent-state ratio (this deployment pins no override). Real capacity is bounded by both pools at once and must be read from the engine's own reported budget at deployment, not from this arithmetic.

### 3. Concurrent 128k Context Session Capacity

For an active context window of 128,000 tokens (128k), assuming FP8 KV cache quantization with Kimi Delta Attention compression. **Note that FP8 KV is not the shipped default** — `SGLANG_KV_CACHE_DTYPE` is empty, matching the cookbook's B200 cell, so these figures apply only if you opt in by setting it to `fp8_e4m3`:

$$\text{KV Footprint per 128k Session} \approx 26.9\,\text{GB}$$

$$\text{Max Concurrent 128k Sessions per Replica} = \left\lfloor \frac{887\,\text{GB}}{26.9\,\text{GB}} \right\rfloor = 32\text{ concurrent streams}$$
*Estimate, pending release.*

$$\text{Concurrent 128k Sessions per GPU} \approx 2.0\text{ streams/GPU}$$
*Estimate, pending release.*

> [!NOTE]
> **KDA Recurrent State vs. KV Caching Caveat:** The KDA recurrent-state component across Kimi K3's 69 linear attention layers makes per-session memory far flatter in context length than pure-attention models. While the $\approx 26.9\,\text{GB}$ per 128k session derivation estimates roughly 32 concurrent sessions per replica, first-deployment measurement supersedes the math.

### 4. Cluster Capacity Scaling Reference Table (DP=N Replicas)

While **1x 2-Node Replica (DP=1, 16x B200 GPUs) serves as the turnkey MVP baseline**, the architecture scales out horizontally to N replicas via Data Parallelism (DP=N) over ReadOnlyMany Hyperdisk ML:

| Cluster Scale (DP=N Replicas) | Physical Nodes (`a4-highgpu-8g`) | Total B200 GPUs | Total HBM3e Memory | Dedicated KV Cache Pool | Max Concurrent 128k Streams (*estimate, pending release*) | Aggregate Output Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1x Replica (DP=1, MVP Baseline)** | 2 Nodes | 16 | 2,880 GB (180 GB/GPU) | 887 GB | ~32 sessions | 1.0x (R<sub>base</sub>) |
| **2x Replicas (DP=2)** | 4 Nodes | 32 | 5,760 GB | 1,774 GB | ~64 sessions | 2.0x (2 R<sub>base</sub>) |
| **4x Replicas (DP=4)** | 8 Nodes | 64 | 11,520 GB | 3,548 GB | ~128 sessions | 4.0x (4 R<sub>base</sub>) |
| **Nx Replicas (DP=N)** | 2N Nodes | 16N | N x 2,880 GB | N x 887 GB | N x 32 sessions | Nx (N R<sub>base</sub>) |

> **Per-GPU Normalization Rule:** In all benchmark normalization calculations and performance reporting, aggregate cluster throughput must be divided by **`16.0`** (for a 1-replica 2-node pool) rather than 8.0, reflecting the 16x B200 GPUs required to host Kimi K3's 2.8T parameter footprint.

> [!NOTE]
> **Horizontal Autoscaling Absence:** Unlike reference stacks such as GLM, Kimi K3 serves as a single 2-node MPI replica across 16 B200s (`SERVING_REPLICAS=1`, `NODES_PER_REPLICA=2`, and `GPU_MAX_NODES=2`), leaving no spare GPU capacity in the cluster to scale into. Furthermore, dynamic pod scale-out events would require re-hydrating 1,453.7 GiB of model weights from GCS or local disk. Horizontal Pod Autoscaling (HPA) is therefore intentionally not implemented in this repository.

> [!NOTE]
> **Spot Provisioning Resilience:** Reclamation of ANY GPU in the 2-node TP16/EP16 group takes down the whole serving replica until a replacement joins; spot suits benchmarking, on-demand is recommended for production.

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

Truncating or stripping thought blocks during multi-turn agentic execution causes catastrophic reasoning degradation and context misalignment. The gateway guarantees end-to-end transmission of raw reasoning tokens to client applications and BigQuery audit logs. Note that using the older `kimi_k2` parser value silently leaks chain-of-thought into `content`; this repo pins `kimi_k3`.

In multi-turn agentic workflows, client applications must echo back the full assistant message including reasoning (`<think>...</think>`) and tool-call content between turns. While the LiteLLM Gateway preserves thought blocks and guarantees their transmission, the obligation to echo them back in subsequent turn history rests entirely on the client.

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
    -   **Native Parsers**: Configured with `--reasoning-parser kimi_k3 --tool-call-parser kimi_k3` as prescribed in the SGLang cookbook. Using the older `kimi_k2` parser value silently leaks chain-of-thought into `content`; this repo pins `kimi_k3`.
    -   **Inter-Node Interconnect**: SGLang relies on NCCL
        GPUDirect RDMA over RoCEv2 (tuned via GKE gIB `set_nccl_env.sh`) for high-speed inter-host tensor passing
        across the 16x B200 GPUs.
        - *Verified at runtime, not assumed*: with `NCCL_DEBUG=INFO`, the live 2-node deployment reported NCCL `2.28.3` assigning the **gIB** net plugin, auto-detecting the platform as **`a4`** and loading `/usr/local/gib/configs/tuner_config_a4.txtpb` — the A4 profile, not the A4X one this repo explicitly rejects above. Ring construction showed cross-node hops as `NET/gIB/<n>/GDRDMA` (GPUDirect RDMA on the wire) and intra-node hops as `P2P/IPC` (NVLink), with `0 nvls channels`, consistent with `enable_nccl_nvls=False`. Each pod binds all eight `networking.gke.io.networks/rdma-0..7` virtual functions the `a4-highgpu-8g` node exposes, one per GPU.
    -   **Cookbook Parity (A4 / B200, 2 nodes)**: The launch command tracks the [SGLang Kimi-K3 Cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) cell for **B200 · 2 nodes**, which is this deployment's exact topology: `--trust-remote-code --tp-size 16 --mem-fraction-static 0.85 --disable-flashinfer-autotune --watchdog-timeout 3600 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --model-loader-extra-config '{"enable_multithread_load": true}'`. The cookbook's `--dcp-size 16` belongs to the *balanced* and *high-throughput* profiles; this repo starts from *low latency* and omits it. Beyond the cookbook the repo adds only what multi-node GKE requires (`--dist-init-addr`, `--dist-timeout 3600`, `--nnodes`, `--node-rank`), observability (`--enable-metrics`), and its own `--context-length`, `--pp-size`, `--ep-size` and `--schedule-policy` knobs.
    -   **No Hand-Picked Kernels**: `SGLANG_PREFILL_ATTENTION_BACKEND`, `SGLANG_DECODE_ATTENTION_BACKEND`, `SGLANG_LINEAR_ATTN_PREFILL_BACKEND`, `SGLANG_MOE_RUNNER_BACKEND` and `SGLANG_KV_CACHE_DTYPE` all ship **empty**, because no Blackwell cell in the cookbook (B200, B300, GB200, GB300) pins any of them. SGLang classifies `KimiK3ForConditionalGeneration` as `AttentionArch.MLA` and dispatches the kernels itself. On the B200 deployment measured here it resolved them as follows, read from the running engine's own startup banner and resolved `server_args` rather than predicted: `attention_backend`, `prefill_attention_backend` and `decode_attention_backend` all became **`trtllm_mla`** (announced as *"Use trtllm_mla as the default prefill and decode attention backend for Kimi-K3 on SM100/SM103"*), `linear_attn_backend` became `triton`, `moe_runner_backend` became `flashinfer_mxfp4`, and `kv_cache_dtype` stayed `auto`. FlashInfer does appear, but as `sampling_backend` — not as an attention kernel. `--decode-attention-backend flashmla` appears in the cookbook **only on the H100 and H200 cells** — it is a Hopper kernel — and `trtllm_mha` is the SM100 default for MHA architectures, which K3 is not. `tests/check_render_exceptions.sh` fails the render if either, or a combined `--attention-backend`, reappears. Note also that RadixAttention prefix-cache reuse benefits only the 24 MLA layers; the 69 KDA linear recurrent-state layers are not prefix-shareable.
    -   **Wrong-Machine Flags Are Rejected**: `--mamba-full-memory-ratio` is not a cookbook flag; the value previously carried here came from `AI-Hypercomputer/gpu-recipes` `a4x/multi-host-serving/sglang`, a **4-node A4X/GB200 NVL72** recipe. A4X is Grace-Blackwell on an NVL72 fabric and A4 is B200 HGX over RoCEv2 — different machines, non-transferable tuning. The render check now treats that flag as forbidden.
    -   **Parallelism Profiles (`SGLANG_PARALLEL_PROFILE`)**: Supports `tp16` (default `--tp-size 16`, confining all parallelism to TP/EP=16 across 16 GPUs) and `tp8pp2` (fallback `--tp-size 8 --pp-size 2` when inter-node RoCEv2 interconnect proves throughput-bound, confining TP all-reduce collectives to NVLink within each node and transferring pipeline activations over RoCE; the profile also drops EP to 8 for the same reason).
        - *When to flip*: Flip to `tp8pp2` if inter-node all-reduce over RoCEv2 is measured as the primary latency bottleneck at first deployment.
        - *Uneven Layer Split Caveat*: Kimi-K3 has `num_hidden_layers = 93` (an odd number), so PP=2 automatic split is uneven by construction. Furthermore, full-attention (MLA) layers occur at every 4th layer plus the last (`text_config.linear_attn_config.full_attn_layers`), causing the two pipeline stages to receive unequal MLA counts and unequal KV-cache memory. Environment variable `SGLANG_PP_LAYER_PARTITION` exists to override the automatic split, and the correct partition is `TBD — to be measured at first deployment`.

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
-   **Measured NCCL Bus Bandwidth — `286.763 GB/s`**: `03_deploy_workloads.sh`
    refuses to deploy the serving engine until an `all_reduce_perf` sweep across
    the full 2-node / 16-GPU topology clears a `100 GB/s` floor. On
    `a4-highgpu-8g` spot nodes in `europe-north1-b` the gate recorded
    **286.763 GB/s** bus bandwidth, after which the serving-image parity gate
    re-ran the same collective inside the actual runtime container and reported
    `NCCL_PARITY_RESULT pass` — confirming the fabric result is a property of the
    cluster and not of the purpose-built test image.
    -   *Gate sequencing*: the two fabric gates run strictly one after the other.
        Each gate pod claims all eight `networking.gke.io.networks/rdma-0..7`
        interfaces and a node advertises exactly one of each, so a node hosts at
        most one gate pod irrespective of its GPU request. Applying both
        2-replica StatefulSets concurrently deadlocks permanently on the
        documented 2-node topology — two pods run and two stay `Pending` until
        both gates time out.

#### Engine Feature & Architecture Comparison Table

Feature / Metric              | SGLang (`sglang`)                                      | NVIDIA TensorRT-LLM (`trtllm`)
:---------------------------- | :----------------------------------------------------- | :----------------
**Status in Repository**      | **Primary Default** (`INFERENCE_ENGINE="sglang"`)      | **Experimental** (No published TRT-LLM K3 support as of launch)
**Container Image**           | `lmsysorg/sglang:kimi-k3`                              | Custom TRT-LLM builder
**Distributed Orchestration** | Native PyTorch Distributed (`--dist-init-addr`, **No Ray**) | OpenMPI (`mpirun -n 16 --hostfile /tmp/hostfile`)
**Inter-Node Networking**     | RoCEv2 GPUDirect RDMA (tuned via GKE gIB `set_nccl_env.sh`) | RoCEv2 GPUDirect RDMA (tuned via GKE gIB `set_nccl_env.sh`)
**Weight & Kernel Format**    | Native `KimiK3ForConditionalGeneration` / MXFP4 / engine-dispatched kernels (no backend pinned, per the cookbook's Blackwell cells) | Direct PyTorch checkpoint loading (Experimental)
**Parsers**                   | `--reasoning-parser kimi_k3 --tool-call-parser kimi_k3` | Custom regex parser
**Prefix Caching**            | RadixAttention (only for 24 MLA layers; 69 KDA layers not prefix-shareable) | Standard KV Cache block reuse (MLA layers only)
**Memory Management**         | `--mem-fraction-static 0.85`                           | `--kv_cache_free_gpu_memory_fraction 0.90`
**Ideal Workload Profile**    | Dynamic interactive sessions, structured JSON, reasoning prompts | Maximum raw throughput batch serving (experimental)

#### Live Benchmark Performance Comparison

<!-- ENGINE_COMPARISON_START -->

### Live Benchmark Performance (SGLang)

All benchmarks were executed on the live GKE serving cluster with identical hardware allocations (16x NVIDIA B200 HGX across 2 nodes, GKE `a4-highgpu-8g` node pool, NVLink 5th-gen, RoCEv2 GPUDirect RDMA fabric) and identical model weights mounted read-only from a shared Hyperdisk ML volume. The engine served via the LiteLLM Enterprise Gateway on port 4000 (Standard, Massive, Soak) and direct container port 8000 (Saturation Sweep, Prefill Ingestion).

**Note:** NVIDIA TensorRT-LLM was not benchmarked in this run, so comparative delta columns and selection guidance are omitted.

#### Methodology & Provenance Protocol
* **Cache Policy:** Workload suites (Standard, Massive, Soak) evaluated end-to-end serving performance on port 4000, where dynamic prompt nonce injection bypassed LiteLLM Redis exact-match caching. The Concurrency Saturation Sweep and Prefill Ingestion suites evaluated direct engine performance on port 8000, where every request carries a 16-character random nonce in its leading prompt tokens so that no two requests share a radix-cache prefix, ensuring 0% prefix-cache hits (measuring true cold decoding and prefill throughput). No engine cache-flush API is invoked; prefix reuse is defeated by construction rather than by an out-of-band flush.
* **Sequential Execution & Drain Protocol:** Suites never overlap. Each runs as a single Kubernetes Job, and the next Job is only created once the previous one has reported completion, so the engine has drained every in-flight request before the following suite issues its first. This is enforced rather than assumed: the provenance gate below rejects a result set whose suite intervals overlap or run out of order, and `tests/adv_audit_benchmark_integrity.py` re-derives the same check in CI from each suite's recorded start timestamp and measured duration.
* **Engine Provenance Verification:** Engine identity was taken from the running deployment before every suite rather than from the benchmark's own configuration. `scripts/05_run_benchmarks.sh` reads the version by executing `import sglang; sglang.__version__` (or `tensorrt_llm.__version__`) inside the serving container, and reads the image reference from the running pod spec; both are stamped into every result file's `metadata` block. A suite whose recorded engine, image or version is missing, mismatched or a placeholder is refused publication by the provenance gate in this script. Collection timestamps recorded in suite metadata:
  * **SGLang** (`sglang-blackwell:latest`): Standard (2026-08-01T00:04:47Z), Massive (2026-08-01T00:05:45Z), Soak (2026-08-01T00:07:55Z), Saturation (2026-08-01T00:38:38Z), Prefill (2026-08-01T01:11:28Z).

#### Table 1: Production Workload Suite Summary (Gateway Port 4000)
| Workload Suite | Metric | SGLang (0.5.16) |
| :--- | :--- | :--- |
| Standard Suite ($c=8$, $128\text{ tok}$) | TTFT P50 (ms) | 632.21 |
|  | TPOT mean (ms) | 30.71 |
|  | Throughput (tok/s) | 225.40 |
|  | Success rate | 100.0% |
| Massive Stress ($c=20$, $256\text{ tok}$) | TTFT P50 (ms) | 903.47 |
|  | TPOT mean (ms) | 35.53 |
|  | Throughput (tok/s) | 244.84 |
|  | Success rate | 100.0% |
| Endurance Soak ($c=18$, $1800\text{s}$) | TTFT P50 (ms) | 614.75 |
|  | TPOT mean (ms) | 40.68 |
|  | Throughput (tok/s) | 420.54 |
|  | Completed cycles | 2981 |

#### Table 2: ISL/OSL x Concurrency Saturation Sweep (Direct Port 8000, 0% Cache Hits)
| Grid Cell (ISL/OSL, $c$) | Prompt tok (measured) | SGLang (0.5.16) tok/s | SGLang (0.5.16) TTFT P99 (s) |
| :--- | :--- | :--- | :--- |
| $1k/1k$, $c=1$ | 918 | 40.52 | 0.3700 s |
| $1k/1k$, $c=8$ | 917 | 258.51 | 0.7155 s |
| $1k/1k$, $c=32$ | 918 | 840.23 | 2.7551 s |
| $1k/1k$, $c=128$ | 918 | 2314.46 | 5.4815 s |
| $8k/1k$, $c=1$ | 7,133 | 40.29 | 0.4386 s |
| $8k/1k$, $c=8$ | 7,134 | 241.62 | 2.8154 s |
| $8k/1k$, $c=32$ | 7,134 | 684.28 | 9.9412 s |
| $8k/1k$, $c=128$ | 7,134 | 1152.47 | 94.4919 s |
| $32k/2k$, $c=1$ | 28,445 | 39.46 | 1.3314 s |
| $32k/2k$, $c=8$ | 28,446 | 220.14 | 10.1987 s |
| $32k/2k$, $c=32$ | 28,446 | 427.97 | 124.8238 s |
| $32k/2k$, $c=128$ | — | *SKIPPED* | *4,194,304 in-flight prompt tokens exceeds MAX_INFLIGHT_PROMPT_TOKENS=2,000,000* |
| $128k/2k$, $c=1$ | — | *SKIPPED* | *ISL+OSL=133,120 tokens exceeds the engine context window MAX_CONTEXT_TOKENS=131,072; the engine rejects such requests with HTTP 400 before any tokens are generated* |
| $128k/2k$, $c=8$ | — | *SKIPPED* | *ISL+OSL=133,120 tokens exceeds the engine context window MAX_CONTEXT_TOKENS=131,072; the engine rejects such requests with HTTP 400 before any tokens are generated* |
| $128k/2k$, $c=32$ | — | *SKIPPED* | *ISL+OSL=133,120 tokens exceeds the engine context window MAX_CONTEXT_TOKENS=131,072; the engine rejects such requests with HTTP 400 before any tokens are generated* |
| $128k/2k$, $c=128$ | — | *SKIPPED* | *ISL+OSL=133,120 tokens exceeds the engine context window MAX_CONTEXT_TOKENS=131,072; the engine rejects such requests with HTTP 400 before any tokens are generated* |

> The ISL in each cell label is the *requested* input length. The measured column is the prompt length the tokenizer actually produced for that cell, as reported by the server's `usage.prompt_tokens`, and is the length the throughput and TTFT figures beside it were obtained at. The sweep builds each prompt by repeating a fixed synthetic passage `round(ISL / BASE_TOKENS_APPROX)` times, so a cell reaches its target only as accurately as that constant describes the passage. This run used `BASE_TOKENS_APPROX`=1,024, recorded in the result file's `grid` block. Every measured cell above landed at 87%-90% of its labelled ISL. Read the measured column, not the label, when comparing against another system.

#### Table 3: Prompt Prefill Ingestion Stress ($5,777\text{ prompt tok measured} \to 16\text{ out}$)
| Metric | SGLang (0.5.16) |
| :--- | :--- |
| Prefill throughput | 15502.49 prompt tok/s |
| TTFT mean (ms) | 372.65 ms |

<!-- ENGINE_COMPARISON_END -->

---

## 📦 Hybrid 3-Tier Storage Co-Design & Weight Backup Lifecycle

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
|                     Tier 2: Local NVMe SSD RAID 0 Scratch Array (`/mnt/scratch`)                  |
|  - Capacity: 32x 375 GiB Local NVMe SSDs = 11.7 TiB/node | Formatted XFS via RAID 0 DaemonSet     |
|  - Content: Ultra-fast local buffer for MPI shared memory exchange and runtime staging logs       |
+---------------------------------------------------------------------------------------------------+
                                                  ^
                                                  | (High-Speed Backup Hydration @ measured 2.1 GiB/s)
+---------------------------------------------------------------------------------------------------+
|                   Tier 3: Cloud Storage (GCS) Weight Backup Bucket                                |
|  - URI: `gs://<project>-kimi-k3-weights-backup` | Cost: ~$40/month (outside TF state)             |
|  - Content: Persistent backup of verified weight shards (measured: 11 min 18 s @ 2.1 GiB/s)       |
+---------------------------------------------------------------------------------------------------+
```

### Weight Backup Hydration Lifecycle

When deploying a fresh cluster or recovering from a disaster, downloading 1,453.7 GiB of weights from external model hubs can take hours. Using a pre-populated GCS weight backup bucket (`Tier 3`), the automated hydration job (`02-hydrate-weights-gcs.yaml.template`) transfers the entire 1,453.7 GiB weight footprint into the 2,000 GB Hyperdisk ML volume in **11 min 18 s at an average of 2.1 GiB/s**, measured end to end on this deployment: all 96 shards copied from `gs://${PROJECT_ID}-kimi-k3-weights-backup`, with `gcloud storage rsync` reporting the throughput itself (1,453.7 GiB / 678 s = 2.15 GiB/s, which agrees).

> **The hyperdisk is not what limits this.** The volume was provisioned at the default `hyperdisk_ml_throughput_mibps = 24,576 MiB/s` and hydration sustained roughly 2,150 MiB/s — about 9% of it. The ceiling here is the GCS read path and rsync's parallelism, not the disk, so raising provisioned throughput will not make hydration proportionally faster and the 6,144 MiB/s fallback (used when regional quota is unavailable) still sits comfortably above what the transfer actually draws. Provisioned throughput matters for the *serving* read pattern — 16 ranks faulting in shards concurrently at pod start — which is a different workload from this one-time sequential fill.

#### Operational Seeding & Hydration Commands

```bash
# 1. Standard deploy leveraging an existing GCS weight backup bucket
export GCS_WEIGHTS_BUCKET="gs://YOUR_PROJECT_ID-kimi-k3-weights-backup"
./scripts/03_deploy_workloads.sh

# 2. Automatically populate / seed the GCS weight backup after an initial Hugging Face download
export POPULATE_WEIGHTS_CACHE="true"
export GCS_WEIGHTS_BUCKET="gs://YOUR_PROJECT_ID-kimi-k3-weights-backup"
./scripts/03_deploy_workloads.sh

# 3. Force re-staging of model weight shards (flips volume back to ReadWrite mode temporarily)
export FORCE_WEIGHT_JOB="true"
./scripts/03_deploy_workloads.sh
```

#### Complete Teardown & State Bucket Retention

By default, `./scripts/06_destroy_all.sh` and `terraform destroy` remove all Terraform-managed resources—including the GKE cluster, RoCEv2 network, Cloud SQL instance, and Hyperdisk ML ReadOnlyMany (`ROX`) volume—to ensure zero ongoing cloud spend. For temporary overnight pauses that retain weights and disks, use the scheduled evening turndown CronJob or scale the serving StatefulSet to 0.

There are two GCS buckets retained after `./scripts/06_destroy_all.sh`: the out-of-band Terraform remote state bucket (`gs://${PROJECT_ID}-kimi-k3-tfstate`) and the out-of-band weight backup bucket (`gs://${PROJECT_ID}-kimi-k3-weights-backup`), both listed in an **OUT-OF-BAND RETAINED BUCKET INVENTORY** report. To explicitly delete retained buckets when no longer needed:

```bash
# Delete the out-of-band Terraform remote state bucket
gcloud storage rm --recursive "gs://${PROJECT_ID}-kimi-k3-tfstate"
```

--------------------------------------------------------------------------------

## 📁 Repository Directory Structure

```
kimi-k3-gcp-enterprise-serving/
├── .github/                       # CI workflow (shellcheck, secret scan, render + dependency gates) and Dependabot config
├── LICENSE                        # Apache-2.0 License
├── README.md                      # Reference architecture and operational runbook (this file)
├── benchmarks/                    # Synthetic load testing and stress benchmark Python suites
│   ├── benchmark_kimi_k3.py       # Standard enterprise performance benchmark (TTFT, TPOT, Throughput)
│   ├── generate_comparison.py     # Parity validator; regenerates the engine comparison table between the README markers
│   ├── massive_benchmark_kimi_k3.py # High-concurrency stress test simulating 20 agent streams
│   ├── run_prefill_benchmark_kimi_k3.py # Empirical prompt-ingestion prefill benchmark (8k-in/16-out)
│   ├── run_saturation_sweep_kimi_k3.py # Direct engine saturation sweep across concurrency levels c=1..64
│   ├── soak_benchmark_kimi_k3.py  # 30-minute continuous stability endurance test (1,800 seconds)
│   └── telemetry_sanitizer.py     # Redacts API keys and sensitive fields from telemetry before publication
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
├── terraform/                     # Modular, enterprise-grade Terraform infrastructure definitions
│   ├── main.tf                    # Root composition module integrating cluster, network, and storage modules
│   ├── variables.tf / outputs.tf  # Global input variables and cluster/storage/database outputs
│   ├── terraform.tfvars.example   # Example variables template
│   ├── modules/                   # Infrastructure modules: cluster, network, node_pool_spot, storage,
│   │                              # database, cache, audit, gateway_iam, observability
│   └── manifests/                 # Kubernetes YAML templates and generated runtime manifests
└── tests/                         # Offline suites: tiered cases t1-t5, adversarial gates, secret scan,
                                   # render exceptions, dependency floors, NCCL parse and provenance checks
```

---

## 🛡️ Failover Mechanics, Dynamic Quotas & Self-Healing Resilience

### 1. K8s Readiness Ejection & LiteLLM Router Retries

The Tier 1 Enterprise AI Gateway and Kubernetes infrastructure utilize active health probing via readiness probes and LiteLLM router retries. If a serving pod experiences GPU ECC faults, NVLink lockups, or network drops, Kubernetes readiness probes fail and eject the pod from the upstream ClusterIP routing pool, and LiteLLM retries the failed request.

> [!WARNING]
> **At the shipped default this provides no redundancy.** The stack deploys a single 2-node replica (`SERVING_REPLICAS=1`, `GPU_MAX_NODES=2`), so there is no second healthy replica to retry against — readiness ejection simply empties the routing pool and requests fail until the replica recovers. Retry-to-a-healthy-peer only becomes meaningful at `DP≥2` (4+ GPU nodes). Failover has not been exercised on a live deployment.

### 2. Dynamic Spot Preemption Priorities (P0 > P1 > P2)

To achieve 55%+ compute cost savings, serving compute pools utilize Google Cloud Blackwell Spot VMs (`a4-highgpu-8g`). To prevent cluster-wide outages during spot preemption events, workloads are scheduled across a strict 3-tier preemption hierarchy:

* **Priority P0 (Non-Preemptible System Core):** The Enterprise AI Gateway (LiteLLM), Cloud SQL Auth Proxy, and Redis instance run exclusively on a dedicated, non-preemptible e2-standard system node pool (`np-system`). Control-plane routing and authentication never go offline.
* **Priority P1 (Graceful Serving Workloads):** Serving pods run on the B200 spot pool with Kubernetes `SIGTERM` handling and `preStop` drain hooks. When GCP issues a 30-second spot preemption notice, the pod traps `SIGTERM`, executes the `preStop` drain hook to deregister from the LiteLLM gateway, finishes inflight requests, and terminates cleanly.
* **Priority P2 (Batch Benchmarks Only):** Off-peak massive benchmark suites run at lowest priority, yielding resources instantly to P1 serving pods during quota contention. Reclamation of any GPU in the 2-node TP16/EP16 group takes down the whole serving replica until a replacement joins; spot suits benchmarking, on-demand is recommended for production.

### 3. Warm Pod ROX Recovery (measured: 17 min 03 s)

When a preempted B200 spot node is replaced by the GKE Cluster Autoscaler, traditional architectures suffer multi-hour cold starts downloading weights. Because Kimi K3's weights are mounted via ReadOnlyMany (`ROX`) Hyperdisk ML (the volume holds weights only; TRT-LLM PyTorch backend loads checkpoints directly without pre-compiled engine files), new pods bypass network downloads entirely. Once physical nodes pass CUDA initialization, the 2-node serving replica reaches `Ready` state in ***17 min 03 s***.

Measured on a 2-node `a4-highgpu-8g` spot replica in `europe-north1-b` against an already-hydrated ROX volume, SGLang `tp16` profile:

| Milestone | Timestamp (UTC) | Elapsed |
| :--- | :--- | :--- |
| Node pool scale-up requested (0 → 2 nodes) | 23:13:22 | — |
| Both B200 nodes registered `Ready` with kubelet | 23:25:57 | 12 min 35 s |
| `kimi-k3-serving-0` pod created | 23:36:27 | — |
| Pod scheduled and containers initialized | 23:36:51 | 24 s |
| SGLang reports *"The server is fired up and ready to roll!"* | 23:53:21 | 16 min 54 s |
| Pod condition `Ready=True` | 23:53:30 | **17 min 03 s** |

The 17-minute figure is checkpoint load plus distributed initialization only — no weight transfer occurs, which is precisely what the ROX design buys. Note that node provisioning (12 min 35 s) and pod warm-up (17 min 03 s) are *sequential* on a cold cluster but fully overlapped on a spot replacement, since the ROX volume stays attached to the surviving node. The `startupProbe` budget (`periodSeconds: 30 × failureThreshold: 120`) allows 60 minutes, leaving substantial headroom over the measured value.

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
|  - Execute GCS Weight Hydration Job (measured 11 min 18 s) or Hugging Face Secure Downloader                    |
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
4.  Supports `PURGE_WEIGHTS_BACKUP=true` (when set alongside `FORCE_DESTROY=true`) for explicit pre-destroy weight backup bucket purge (out-of-band bucket is not destroyed by `terraform destroy`).

### Retained Storage & Bucket Purge Guide

`./scripts/06_destroy_all.sh` executes `terraform destroy`, which deletes all Terraform-managed resources including the Hyperdisk ML volume. There are now **two** retained out-of-band buckets: the Terraform remote state bucket (`gs://${PROJECT_ID}-kimi-k3-tfstate`) and the weight backup bucket (`gs://${PROJECT_ID}-kimi-k3-weights-backup`), which are listed in an **OUT-OF-BAND RETAINED BUCKET INVENTORY** report. To completely remove retained storage:

```bash
# 1. Purge and delete the Terraform remote state bucket
gcloud storage rm --recursive "gs://${PROJECT_ID}-kimi-k3-tfstate"

# 2. (Optional) Delete out-of-band GCS weight backup bucket if created
gcloud storage rm --recursive "gs://${PROJECT_ID}-kimi-k3-weights-backup"
```

> [!NOTE]
> The trajectory bucket (`gs://${PROJECT_ID}-kimi-k3-trajectories`) is **not** listed above because it is the one Terraform-managed bucket in the stack (`terraform/modules/storage/main.tf`), so `terraform destroy` already removes it. Only the two out-of-band buckets above survive teardown.
