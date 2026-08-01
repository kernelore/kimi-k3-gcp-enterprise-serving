# KIMI K3 Enterprise Inference Architecture

[![Google Cloud](https://img.shields.io/badge/Google_Cloud-Blackwell_B200-4285F4?style=flat-square&logo=googlecloud&logoColor=white)](https://cloud.google.com/compute/docs/gpus)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-MXFP4_MoE-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/)
[![TensorRT-LLM](https://img.shields.io/badge/Inference-TensorRT__LLM_Experimental-8A2BE2?style=flat-square)](https://github.com/NVIDIA/TensorRT-LLM)
![SGLang](https://img.shields.io/badge/Inference-SGLang_Kimi__K3-8A2BE2?style=flat-square)
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
**SGLang multi-node RoCEv2 serving (Primary Default)** with `--reasoning-parser kimi_k3 --tool-call-parser kimi_k3`. An experimental **NVIDIA TensorRT-LLM MPI** configuration is maintained in the codebase, though no published TensorRT-LLM K3 support exists as of launch.

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
|  - Token-Bucket Rate Limiting (TPM / RPM) & Exact-Match Prompt Caching on Redis (measured 0.40 ms p50 GET)      |
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
| **Cloud Memorystore for Redis** | `module.cache` | In-memory tier for exact-match prompt caching (measured in-VPC from the gateway pod against Redis 6.2.14: `PING` 0.31 ms p50 / 0.67 ms p99, 512 B `GET` 0.40 ms p50 / 0.82 ms p99, 512 B `SET` 0.37 ms p50 / 0.74 ms p99 — sub-millisecond, not the single-digit-ms this previously estimated) and gateway token-bucket rate limiting (RPM/TPM). Note: exact-match prompt caching applies at the gateway level; inside the serving engine, RadixAttention prefix caching reuse benefits only the 24 MLA attention layers, while KDA linear layers are not prefix-shareable. |
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

> [!NOTE]
> **What the engine actually reported.** Taking the paragraph above at its word, the running deployment was asked. `/get_server_info` returned **`max_total_num_tokens = 927,808`** — the admission budget the scheduler enforces, spanning both pools at once, which is the number to plan against rather than the 887 GB derived above.
>
> That budget is reachable. During the saturation sweep the engine logged `KV cache pool is full. Retract requests.` in exactly two cells — **8k ISL @ c=128** (8 events) and **32k ISL @ c=32** (2 events), the two heaviest cells that ran to completion. It recovered by retracting and requeueing rather than erroring, so those runs are valid, but their throughput includes the cost of the retraction. The `32k @ c=128` and all `128k` cells were skipped by the harness before reaching this point and are marked `SKIPPED` in Table 2. Read the two flagged cells as measurements of a pool at its limit, not of one with headroom.

### 3. Concurrent 128k Context Session Capacity

For an active context window of 128,000 tokens (128k), assuming FP8 KV cache quantization with Kimi Delta Attention compression. **Note that FP8 KV is not the shipped default** — `SGLANG_KV_CACHE_DTYPE` ships empty, so these figures apply only if you opt in by setting it to `fp8_e4m3`:

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
        16`. Pinned to `lmsysorg/sglang:kimi-k3`.
    -   **Native Parsers**: Configured with `--reasoning-parser kimi_k3 --tool-call-parser kimi_k3`. Using the older `kimi_k2` parser value silently leaks chain-of-thought into `content`; this repo pins `kimi_k3`.
    -   **Inter-Node Interconnect**: SGLang relies on NCCL
        GPUDirect RDMA over RoCEv2 (tuned via GKE gIB `set_nccl_env.sh`) for high-speed inter-host tensor passing
        across the 16x B200 GPUs.
        - *Verified at runtime, not assumed*: with `NCCL_DEBUG=INFO`, the live 2-node deployment reported NCCL `2.28.3` assigning the **gIB** net plugin, auto-detecting the platform as **`a4`** and loading `/usr/local/gib/configs/tuner_config_a4.txtpb` — the tuner profile matching this hardware. Ring construction showed cross-node hops as `NET/gIB/<n>/GDRDMA` (GPUDirect RDMA on the wire) and intra-node hops as `P2P/IPC` (NVLink), with `0 nvls channels`, consistent with `enable_nccl_nvls=False`. Each pod binds all eight `networking.gke.io.networks/rdma-0..7` virtual functions the `a4-highgpu-8g` node exposes, one per GPU.
    -   **Launch Flags (A4 / B200, 2 nodes)**: The engine is launched with `--trust-remote-code --tp-size 16 --mem-fraction-static 0.85 --disable-flashinfer-autotune --watchdog-timeout 3600 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --model-loader-extra-config '{"enable_multithread_load": true}'`, plus what multi-node GKE requires (`--dist-init-addr`, `--dist-timeout 3600`, `--nnodes`, `--node-rank`), observability (`--enable-metrics`), and this repo's own `--context-length`, `--pp-size`, `--ep-size` and `--schedule-policy` knobs. `--dcp-size 16` is deliberately omitted: it belongs to throughput-oriented profiles, and this deployment starts from a low-latency baseline.
    -   **No Hand-Picked Kernels**: `SGLANG_PREFILL_ATTENTION_BACKEND`, `SGLANG_DECODE_ATTENTION_BACKEND`, `SGLANG_LINEAR_ATTN_PREFILL_BACKEND`, `SGLANG_MOE_RUNNER_BACKEND` and `SGLANG_KV_CACHE_DTYPE` all ship **empty**. SGLang classifies `KimiK3ForConditionalGeneration` as `AttentionArch.MLA` and dispatches the kernels itself; pinning one by hand only risks overriding a better choice the engine would have made. On the B200 deployment measured here it resolved them as follows, read from the running engine's own startup banner and resolved `server_args` rather than predicted: `attention_backend`, `prefill_attention_backend` and `decode_attention_backend` all became **`trtllm_mla`** (announced as *"Use trtllm_mla as the default prefill and decode attention backend for Kimi-K3 on SM100/SM103"*), `linear_attn_backend` became `triton`, `moe_runner_backend` became `flashinfer_mxfp4`, and `kv_cache_dtype` stayed `auto`. FlashInfer does appear, but as `sampling_backend` — not as an attention kernel. Two kernels are explicitly forbidden: `flashmla` is Hopper-only (H100/H200), and `trtllm_mha` is the SM100 default for MHA architectures, which K3 is not. `tests/check_render_exceptions.sh` fails the render if either, or a combined `--attention-backend`, reappears. Note also that RadixAttention prefix-cache reuse benefits only the 24 MLA layers; the 69 KDA linear recurrent-state layers are not prefix-shareable.
    -   **Non-Transferable Tuning Is Rejected**: `--mamba-full-memory-ratio` is not part of this deployment's flag set. The value previously carried here was lifted from a multi-host recipe written for a different machine class and a different interconnect fabric, where it does not transfer to B200 HGX over RoCEv2. `tests/check_render_exceptions.sh` now treats that flag as forbidden.
    -   **Parallelism Profiles (`SGLANG_PARALLEL_PROFILE`)**: Supports `tp16` (default `--tp-size 16`, confining all parallelism to TP/EP=16 across 16 GPUs) and `tp8pp2` (fallback `--tp-size 8 --pp-size 2` when inter-node RoCEv2 interconnect proves throughput-bound, confining TP all-reduce collectives to NVLink within each node and transferring pipeline activations over RoCE; the profile also drops EP to 8 for the same reason).
        - *When to flip*: Flip to `tp8pp2` if inter-node all-reduce over RoCEv2 is measured as the primary latency bottleneck at first deployment.
        - *Uneven Layer Split Caveat*: Kimi-K3 has `num_hidden_layers = 93` (an odd number), so PP=2 automatic split is uneven by construction. Furthermore, full-attention (MLA) layers occur at every 4th layer plus the last (`text_config.linear_attn_config.full_attn_layers`), causing the two pipeline stages to receive unequal MLA counts and unequal KV-cache memory. Environment variable `SGLANG_PP_LAYER_PARTITION` exists to override the automatic split, and the correct partition remains unmeasured. Determining the optimum requires sweeping partitions against a fixed workload — left as future work rather than guessed at here.
        - *Custom All-Reduce Must Be Disabled Under PP=2 (`SGLANG_DISABLE_CUSTOM_ALL_REDUCE`)*: On `tp16` the TP group spans both nodes, so SGLang disables its custom all-reduce automatically and every collective goes through NCCL. On `tp8pp2` the TP group fits inside one node, which re-enables the custom all-reduce path — and that path fails on this deployment during CUDA graph capture with `Capture cuda graph failed: invalid argument` raised from `custom_all_reduce.cuh:508`, inside `register_graph_buffers` → `get_graph_buffer_ipc_meta`. The failure is in CUDA IPC handle exchange for the graph buffers, not memory pressure: it reproduced identically at `--mem-fraction-static` 0.85 and 0.80, and `/dev/shm` is a 512 GiB tmpfs, so neither is the constraint. Setting `SGLANG_DISABLE_CUSTOM_ALL_REDUCE=true` appends `--disable-custom-all-reduce`, falling back to NCCL for the intra-node all-reduce while **keeping CUDA graphs enabled** — deliberately not `--disable-cuda-graph`, which would trade a working collective for a much larger decode regression. With that single flag the `tp8pp2` profile captures all 18 decode graphs in 212 s and serves.

    -   **DSPARK Speculative Decoding (`SGLANG_SPECULATIVE_ALGORITHM`, optional)**: Kimi-K3 ships with a purpose-built speculative decoding algorithm, **DSPARK**. It pairs the full model with a small draft checkpoint ([`RadixArk/Kimi-K3-DSpark`](https://huggingface.co/RadixArk/Kimi-K3-DSpark), 4.19 GiB in a single `model.safetensors` shard) that proposes a block of tokens per step for the target model to verify in one batched forward pass, converting several sequential decode steps into one. It is **opt-in and off by default**: leaving `SGLANG_SPECULATIVE_ALGORITHM` empty renders the launch command byte-identical to the non-speculative one.
        - *Enabling it*: set `SGLANG_SPECULATIVE_ALGORITHM="DSPARK"`. The template then appends `--speculative-algorithm DSPARK --speculative-draft-model-path ${SGLANG_SPECULATIVE_DRAFT_MODEL_PATH}` plus, when set, `--speculative-dspark-block-size ${SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE}` (default `7`) and `--enable-linear-replayssm-spec`. The last flag is K3-specific: the 69 KDA linear-attention layers carry recurrent state, and verifying a rejected draft block requires replaying that state rather than simply discarding KV entries.
        - *Topology*: DSPARK runs on the existing **2 x A4 (16 x B200) TP=16** topology — no extra nodes, no parallelism change. The draft model is small enough that it adds no parallelism decision: it is replicated per rank, not sharded.
        - *Draft Checkpoint Staging*: the draft is **not** placed on the ROX Hyperdisk ML volume. That volume is read-only at serving time, and re-running hydration to add a 4 GiB file would put the 1.4 TiB base checkpoint at risk for no benefit. Instead each serving pod runs a `dspark-draft-fetch` initContainer that `gcloud storage rsync`s `${GCS_WEIGHTS_BUCKET}/${DSPARK_DRAFT_DIR_NAME}` into a pod-local `emptyDir` mounted at `/mnt/draft` — about ten seconds at the ~450 MiB/s this bucket sustains, and every pod needs its own copy regardless. The step exits immediately when `SGLANG_SPECULATIVE_ALGORITHM` is empty, so the non-speculative deployment pays nothing but a container start. The draft repository ships **without** a `generation_config.json` and SGLang will not load it without one, so the initContainer copies the base model's file from the ROX mount into the draft directory rather than leaving that to an operator's memory.
        - *Fail-Closed Gates*: two checks refuse to start a silently-degraded engine. The `dspark-draft-fetch` initContainer aborts if the GCS draft prefix is absent or empty, and its `DSPARK DRAFT GATE` asserts at least one safetensors shard plus a `config.json` before reporting shard count and size. The serving container then re-checks, at launch, that the draft directory exists and contains `generation_config.json`, exiting non-zero with a remediation line instead of falling back to non-speculative decoding without saying so.

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
**Weight & Kernel Format**    | Native `KimiK3ForConditionalGeneration` / MXFP4 / engine-dispatched kernels (no backend pinned) | Direct PyTorch checkpoint loading (Experimental)
**Parsers**                   | `--reasoning-parser kimi_k3 --tool-call-parser kimi_k3` | Custom regex parser
**Prefix Caching**            | RadixAttention (only for 24 MLA layers; 69 KDA layers not prefix-shareable) | Standard KV Cache block reuse (MLA layers only)
**Memory Management**         | `--mem-fraction-static 0.85`                           | `--kv_cache_free_gpu_memory_fraction 0.90`
**Ideal Workload Profile**    | Dynamic interactive sessions, structured JSON, reasoning prompts | Maximum raw throughput batch serving (experimental)

#### Recommended Configuration — the Shipped Default, and the Basis for Every Comparison

**This is the configuration to run, and it is the one that produced the best
numbers in this README.** It is what `scripts/config.env.example` ships unmodified,
what every table below was measured on, and the only configuration used in the
external cross-engine comparison. Everything else documented here — `TP8/PP2`,
DSPARK speculative decoding, FP8 KV cache — is an **optional overlay that is off by
default**, kept in its own section so it is never mistaken for the baseline.

| Setting | Default value | Rationale |
| :--- | :--- | :--- |
| `INFERENCE_ENGINE` | `sglang` | The only engine with published Kimi-K3 support today. |
| `GPU_MACHINE_TYPE` / `GPU_MAX_NODES` | `a4-highgpu-8g` / `2` | 16x B200 HGX, NVLink intra-node, RoCEv2 GPUDirect RDMA inter-node. |
| `SGLANG_PARALLEL_PROFILE` | `tp16` | `--tp-size 16 --pp-size 1 --ep-size 16`; one 16-way tensor group, no pipeline bubbles. |
| `SGLANG_MEM_FRACTION_STATIC` | `0.85` | Largest static fraction that still leaves headroom for CUDA-graph capture. |
| `SGLANG_SCHEDULE_POLICY` | `lpm` | Longest-prefix-match admission. |
| `SGLANG_CONTEXT_LENGTH` | `131072` | Full 128k window. |
| `SGLANG_REASONING_PARSER` / `SGLANG_TOOL_CALL_PARSER` | `kimi_k3` | `kimi_k2` silently leaks chain-of-thought into `content`. |
| All four `*_ATTENTION_BACKEND` / `*_RUNNER_BACKEND` vars | **empty** | The engine resolves better kernels than a hand-pin does (see "No Hand-Picked Kernels"). |
| `SGLANG_KV_CACHE_DTYPE` | **empty** (`auto`) | FP8 is an opt-in overlay, not the baseline. |
| `SGLANG_SPECULATIVE_ALGORITHM` | **empty** | DSPARK off; the launch command renders byte-identical to non-speculative. |
| `SGLANG_ENABLE_TORCH_COMPILE` | `false` | Capture cost is not repaid at these batch sizes. |
| `SGLANG_CHUNKED_PREFILL_SIZE` / `SGLANG_MAX_RUNNING_REQUESTS` | **empty** | Engine defaults; unmeasured knobs are not guessed at. |

Deploy it by copying `scripts/config.env.example` to `scripts/config.env`, filling in
project-specific values, and changing none of the above.

#### Live Benchmark Performance Comparison

<!-- ENGINE_COMPARISON_START -->

### Live Benchmark Performance (SGLang)

**Configuration under test.** SGLang (0.5.16), image `sglang-blackwell:latest`, **TP=16 / PP=1 / EP=16** across 2 x `a4-highgpu-8g` (16x NVIDIA B200 HGX), NVLink 5th-gen intra-node and RoCEv2 GPUDirect RDMA inter-node, MXFP4 MoE weights mounted read-only from a shared Hyperdisk ML volume, engine-dispatched attention kernels, **no speculative decoding**. Workload suites ran through the LiteLLM gateway (port 4000); the saturation sweep and prefill stress ran direct against the engine (port 8000).

**Best measured result: 2,314.46 aggregate output tok/s** at $1k/1k$, $c=128$ on the configuration above. Full grid in Table 2.

Suites never overlap — each is its own Kubernetes Job, started only once the previous has drained. Every prompt carries a 16-character random nonce so no two requests share a radix-cache prefix (0% prefix-cache hits, achieved by construction rather than a cache-flush API). Engine identity is read from the running pod rather than from benchmark config and stamped into every result file; the provenance gate here and `tests/adv_audit_benchmark_integrity.py` reject a set whose suites overlap, run out of order, or carry a missing or placeholder version. NVIDIA TensorRT-LLM was not benchmarked in this run, so delta columns are omitted.

<sub>Collected from `sglang-blackwell:latest` — Standard (2026-08-01T00:04:47Z), Massive (2026-08-01T00:05:45Z), Soak (2026-08-01T00:07:55Z), Saturation (2026-08-01T00:38:38Z), Prefill (2026-08-01T01:11:28Z).</sub>

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

> Cell labels are the *requested* ISL; the measured column is what the tokenizer actually produced (`usage.prompt_tokens`) and is the length each figure was obtained at. Prompts repeat a fixed synthetic passage `round(ISL / 1,024)` times, landing at 87%-90% of the labelled ISL. Compare against the measured column, not the label.

> **Not run** ($32k/2k$, $c=128$): 4,194,304 in-flight prompt tokens exceeds MAX_INFLIGHT_PROMPT_TOKENS=2,000,000

> **Not run** ($128k/2k$, $c=1$, $128k/2k$, $c=8$, $128k/2k$, $c=32$, $128k/2k$, $c=128$): ISL+OSL=133,120 tokens exceeds the engine context window MAX_CONTEXT_TOKENS=131,072; the engine rejects such requests with HTTP 400 before any tokens are generated

#### Table 3: Prompt Prefill Ingestion Stress ($5,777\text{ prompt tok measured} \to 16\text{ out}$)
| Metric | SGLang (0.5.16) |
| :--- | :--- |
| Prefill throughput | 15502.49 prompt tok/s |
| TTFT mean (ms) | 372.65 ms |

<!-- ENGINE_COMPARISON_END -->

### Optional Performance Profiles: TP8/PP2 and DSPARK Speculative Decoding

This repository ships two optional serving profiles on top of the `TP16 / PP1 / EP16` default
measured above. Both were benchmarked on the **same two `a4-highgpu-8g` nodes
(16x B200)**, the same weights mounted from the same read-only Hyperdisk ML
volume, the same direct container port 8000 path, and the same
`benchmarks/run_saturation_sweep_kimi_k3.py` harness. No nodes were added.

* **`TP8 / PP2 / EP8`** — pipeline parallelism across the two nodes instead of a
  16-way tensor group, requiring `SGLANG_DISABLE_CUSTOM_ALL_REDUCE=true` (see
  the tuning notes above).
* **`TP16 / PP1 / EP16 + DSPARK`** — speculative decoding with the
  `RadixArk/Kimi-K3-DSpark` draft model, block size 8. Layered on the **default
  topology**, not on TP8/PP2; the two profiles were not combined.

#### Table 4: Aggregate Output Throughput by Profile (Direct Port 8000, tok/s)

| Grid Cell (ISL/OSL, $c$) | TP16 (default) | TP8/PP2 | TP16 + DSPARK | DSPARK vs default | DSPARK vs TP8/PP2 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| $1k/1k$, $c=8$ | 276.88 | 306.05 | **1058.34** | **3.82x** | 3.46x |
| $1k/1k$, $c=16$ | 492.02 | 527.34 | **1661.36** | **3.38x** | 3.15x |
| $1k/1k$, $c=32$ | 879.49 | 931.21 | **1859.54** | **2.11x** | 2.00x |
| $8k/1k$, $c=8$ | 255.71 | 287.43 | **778.21** | **3.04x** | 2.71x |
| $8k/1k$, $c=16$ | 425.15 | 500.38 | **1014.10** | **2.39x** | 2.03x |
| $8k/1k$, $c=32$ | 681.89 | 829.69 | **1366.15** | **2.00x** | 1.65x |
| $32k/2k$, $c=8$ | 224.27 | 265.83 | **593.19** | **2.64x** | 2.23x |

DSPARK is faster than both alternatives on every cell measured. TP8/PP2 is a
smaller but consistent 1.06x-1.22x gain over the default, widening as ISL grows.
All 21 cells completed with a **0.00% error rate** and identical request counts,
identical generated-token totals (16,384 / 32,768 / 65,536) and 100% success, so
the columns are matched workloads rather than differently-sized runs.

#### Table 5: Per-Stream Latency by Profile

| Grid Cell (ISL/OSL, $c$) | Effective TPOT, TP16 | Effective TPOT, TP8/PP2 | Effective TPOT, DSPARK | TTFT P50, TP16 | TTFT P50, DSPARK |
| :--- | ---: | ---: | ---: | ---: | ---: |
| $1k/1k$, $c=8$ | 28.89 ms | 26.14 ms | **7.56 ms** | 18.91 s | **0.73 s** |
| $1k/1k$, $c=16$ | 32.52 ms | 30.34 ms | **9.63 ms** | 21.98 s | **0.95 s** |
| $1k/1k$, $c=32$ | 36.38 ms | 34.36 ms | **17.21 ms** | 18.59 s | **1.52 s** |
| $8k/1k$, $c=8$ | 31.29 ms | 27.83 ms | **10.28 ms** | 15.12 s | **2.02 s** |
| $8k/1k$, $c=16$ | 37.63 ms | 31.98 ms | **15.78 ms** | 21.21 s | **1.93 s** |
| $8k/1k$, $c=32$ | 46.93 ms | 38.57 ms | **23.42 ms** | 28.57 s | **3.42 s** |
| $32k/2k$, $c=8$ | 35.67 ms | 30.09 ms | **13.49 ms** | 23.66 s | **6.37 s** |

Effective TPOT here is derived as $1000 / \text{per\_user\_tok\_s}$ — the
wall-clock time a single stream spends per generated token, inclusive of its own
queueing and TTFT. It is not the harness's `tpot_ms` field; see the caveat below.

#### Reading These Numbers Honestly

* **Table 4 is a ceiling, not a planning figure — and the penalty was measured, not
  estimated.** The harness builds every prompt by repeating one fixed synthetic
  passage, and a draft model is exceptionally good at continuing text it has already
  seen verbatim. Re-running the same shapes against the same live engine with
  **distinct non-repetitive prompts** (coherent English, ~1,523 tokens each, nothing
  repeated within or across requests) collapses the gain:

  | DSPARK engine, $1k/1k$ | Accepted tok/step, repeated | Accepted tok/step, non-repetitive | tok/s, repeated | tok/s, non-repetitive |
  | :--- | ---: | ---: | ---: | ---: |
  | $c=8$ | 6.41 | **2.20** | 1058.34 | 401.99 |
  | $c=16$ | 6.40 | **2.28** | 1661.36 | 656.60 |
  | $c=32$ | 6.29 | **2.24** | 1859.54 | 1028.57 |

  Acceptance is the mechanism and it is stable: **~2.2 - 2.3 accepted tokens per verify
  step on non-repetitive text against 6.3 - 6.4 on a repeated passage**, a 2.8x collapse
  holding across all three concurrencies (derived from generated tokens over the
  `sglang:spec_verify_calls_total` delta). Effective TPOT at $c=16$ moves from 9.63 ms
  to 24.37 ms. **Plan capacity from the non-repetitive columns** unless the workload
  genuinely is repetitive. Both properties of that right-hand column make it
  conservative rather than flattering — it issues $c$ requests as one burst, so it pays
  more batch-drain penalty than Table 4's $2c$-through-a-$c$-wide-pool design, and its
  prompts are 66% longer. Its 656.60 against the default profile's 492.02 is indicative
  only: the non-speculative baseline could not be re-run on this prompt set before the
  spot pair was torn down, so the two are not a controlled pair.
* **Never quote `tpot_ms` from a speculative-decoding result file.** The harness
  timestamps SSE chunks and treats consecutive chunks as consecutive tokens; under
  speculation one chunk carries every token accepted in a verify step, so the field
  measures **per-step** latency and *rises* as speculation improves. Measured against
  the running DSPARK engine: 512 tokens in 196 chunks (2.61 tok/chunk), inter-chunk P50
  of 29.02 ms against a true 11.84 ms/token — **2.45x** off. Tables 4 and 5 avoid the
  field entirely, deriving from `usage.completion_tokens` and wall-clock duration.
* **The gain narrows as the batch saturates** (3.82x at $c=8$ down to 2.11x at $c=32$ on
  $1k/1k$): a full batch already amortises weight loading, leaving speculation less to
  recover.
* **TTFT is not cross-profile comparable.** These cells burst their requests, so TTFT is
  dominated by admission queueing, not prefill.
* **The $1k/1k$ TP8/PP2 row is a warm re-run** — the cold first cell read 109.29 tok/s
  against 306.05 warm. Every Table 4 figure comes from a warmed engine.
* **A constant 123-token prompt offset** separates the default-profile run from the
  DSPARK run, identical at $1k$, $8k$ and $32k$, so it is chat-template overhead. At the
  worst cell it is 0.2% of run duration (under 1.5% even at a pessimistic 2,000 tok/s
  prefill rate) and cannot account for a 2.00x-3.82x gap.
* **$32k/2k$ at $c=16$ and $c=32$ was not run** on these optional profiles — GPU time on
  the spot pair ran out. Those cells are absent, not omitted for being unfavourable.

<!-- EXTERNAL_COMPARISON_START -->
<!-- Third-party citation block. The legacy-model-string guards (tests/test_cases_t1.sh
     t1_f1_05, tests/adv_test_serving_remediation.sh Check 1) exempt the text between
     these markers so a competing engine can be named in a sourced comparison. The ban
     remains absolute in every other file and in the rest of this README. Prose only --
     no engine may be configured or deployed from inside this block. -->

### External Cross-Engine Reference: Published vLLM Results on an Identical 2 x 8xB200 Topology

The closest public comparison for this deployment is a published NVIDIA Developer Forums benchmark of **Kimi K3 across two NVIDIA 8xB200 nodes (16 GPUs) served with vLLM**: <https://forums.developer.nvidia.com/t/ruuning-kimi-k3-across-two-nvidia-8xb200-nodes-using-vllm/378623>. Same model, same GPU count, same node count, different engine — the only like-for-like external datapoint currently available.

**This repository does not deploy vLLM.** The figures below are quoted from that published run, not reproduced here. NVIDIA TensorRT-LLM remains the planned second engine (see the engine comparison table above) and will be benchmarked in-cluster once K3 support is ready.

**Best measured result against the published run.** On the recommended default
configuration this stack's best measured figure is **2,314.46 aggregate output tok/s**
($1k/1k$, $c=128$, direct to engine, no speculative decoding). The highest output
throughput reported anywhere in the published vLLM run is **378.14 tok/s** ($1k/8k$,
decode-heavy). The 6.12x ratio between them is **not** a like-for-like claim and is not
made here: the published run never issues more than 16 concurrent requests, while
2,314.46 is what these two nodes do when allowed to fill a 128-wide batch. Held at the
published run's own concurrency, the defensible figure is **1.36x - 1.49x on aggregate
throughput and 1.98x on raw decode step time** — that is the comparison the table below
makes, and it is the one to quote.

**Normalising the comparison.** All three published runs use `--num-prompts 16 --request-rate 10000` — concurrency 16 issued as a single burst. The $c=16$ column below is **measured directly at $c=16$**, not interpolated: a dedicated sweep was run on this stack for exactly this comparison.

| Metric at $c=16$ | vLLM (published) | SGLang TP16 (measured) | Delta | SGLang TP16 + DSPARK (measured) | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| $1k/1k$ aggregate output tok/s | 329.64 | **492.02** | **1.49x** | 1661.36 § | 5.04x § |
| $8k/1k$ aggregate output tok/s | 312.63 | **425.15** | **1.36x** | 1014.10 § | 3.24x § |
| Raw decode step (ITL median, ms) | 63.10 | **31.81** | **1.98x faster** | not comparable † | — |
| Effective TPOT (ms) | 31.93 – 44.87 | **31.81** | parity to 1.41x | 9.63 § | 3.32x – 4.66x faster § |
| Prefill ingestion (prompt tok/s) | 12,185 ‡ | **15,502.49** | **1.27x** | not re-measured | — |

§ The DSPARK deltas are **not** claimed as a like-for-like win over the published
vLLM run, and are deliberately left unbolded. Speculative decoding is highly
sensitive to how predictable the generated text is, and this harness's prompts
are a repeated synthetic passage — an unusually favourable case (see the
acceptance-rate caveat above). The published run's prompt distribution is
unknown, and it does not state whether speculative decoding was enabled. The
`TP16` column is the sound comparator; the DSPARK column shows what this
hardware does with speculation on a repetitive workload.

† Under speculative decoding one streamed chunk carries every token accepted in a
verify step, so a chunk-derived inter-token median measures per-step latency and
is not comparable to a non-speculative one. The DSPARK effective-TPOT figure is
derived from generated tokens over wall-clock duration instead. See "Reading
These Numbers Honestly" above.

‡ Derived from the published prompt-heavy run: 129,408 total input tokens / 10.62 s mean TTFT.

The $c=16$ figures come from a `TP16 / PP1 / EP16` sweep issued straight at the serving pod, bypassing the gateway proxy, so that the comparison measures the engine rather than this repository's authentication, budget and caching layer. Two independent $c=16$ runs agreed to within 0.2% (492.49 / 492.02 on $1k/1k$; 424.40 / 425.15 on $8k/1k$). The ITL median on the $8k/1k$ cell is 32.33 ms, essentially unchanged from the 31.81 ms of the $1k/1k$ cell. The DSPARK column is the same sweep re-run on the same two nodes with speculative decoding enabled and nothing else changed; it is an optional profile, opt-in via `SGLANG_SPECULATIVE_ALGORITHM`, and not the default this repository deploys.

The same sweep re-measured the $c=8$ and $c=32$ cells to check that a direct-to-engine run is comparable with the gateway-routed numbers reported in the table above:

| Cell | Gateway-routed (table above) | Direct-to-engine (this sweep) | Difference |
| :--- | :--- | :--- | :--- |
| $1k/1k$, $c=8$ | 258.51 | 276.88 | +7.1% |
| $1k/1k$, $c=32$ | 840.23 | 879.49 | +4.7% |
| $8k/1k$, $c=8$ | 241.62 | 255.71 | +5.8% |
| $8k/1k$, $c=32$ | 684.28 | 681.89 | -0.3% |

The gateway therefore costs at most a few percent of aggregate throughput, and the two measurement bases are interchangeable at the precision quoted here.

**The published run enables speculative decoding; this deployment does not.** Its own reported acceptance statistics are the reason that does not close the gap:

| Published run | Draft tokens | Accepted | Yield | Acceptance length (of 7 drafted) |
| :--- | :--- | :--- | :--- | :--- |
| Prompt-heavy ($8k/1k$) | 55,020 | 8,147 | 14.81% | 2.04 |
| Decode-heavy ($1k/8k$) | 458,073 | 62,560 | 13.66% | 1.96 |
| Balanced ($1k/1k$) | 80,381 | 4,508 | 5.61% | 1.39 |

Per-position acceptance collapses from 24.98% at draft position 0 to 0.12% at position 6 on the balanced run. That deployment spends draft compute on seven tokens to realise 1.39–2.04 of them, and its *raw* step time of ~63 ms is 1.98x this repository's ~32 ms. Speculative decoding recovers most of that deficit but does not overturn it: at its best acceptance the published run reaches effective TPOT parity, and this deployment stays 1.36–1.49x ahead on aggregate throughput without drafting at all.

**Caveats — read before quoting these deltas.**
* **Aggregate deltas likely overstate the steady-state gap.** Sixteen prompts issued as one burst is not steady state: as the batch drains, its tail runs at declining concurrency, which depresses the published aggregate tok/s (total generated / wall duration) and flatters its per-stream TPOT. The **ITL-median** row is the more robust comparator, being far less sensitive to batch drain than a total-over-duration figure. This repository's $c=16$ cells issue 32 requests through a fixed-width 16-worker pool, so the concurrency level is held for the bulk of the run instead of decaying from the first token, and are backed by a 30-minute soak (2,981 cycles, 100% success).
* The published Balanced run's client command specifies `--model nvidia/Qwen3.6-27B-NVFP4` rather than Kimi K3 — the source page is internally inconsistent. A 27B model would not exhibit 44.87 ms TPOT on 16 B200s, so these figures are treated as K3, but that row carries less confidence than the other two.
* The published "peak output tok/s" values (258.00 / 257.00 / 272.00) are *below* their corresponding means (312.63 / 378.14 / 329.64) in all three runs, which is not possible for a true instantaneous peak. That row is not used here.
* The published run states no tensor/pipeline/expert parallelism sizes, no launch command, no vLLM version, no dtype and no interconnect. Hardware equivalence is inferred from "two B200 nodes" alone.
* Do not compare the published prompt-heavy **total token throughput** of 2,841.22 tok/s against this repository's 2,314.46 **output** tok/s — the former counts input tokens. The equivalent total-token figure for the $1k/1k$, $c=128$ cell here is approximately 4,629 tok/s.
* **TTFT is deliberately absent from the comparison table.** The two harnesses attribute admission queueing differently: on a direct-to-engine sweep the scheduler admits a deep client-side burst a couple of requests per prefill round, so a request's measured TTFT includes the wait behind everything admitted before it. That is a property of where the queue sits, not of prefill speed, and it is not comparable against a published TTFT from a different client. Prefill capability is compared through the dedicated prefill suite row instead.

**Status of this section.** The $c=16$ column is measured, not interpolated: a dedicated
sweep was run on this stack for exactly this comparison, and the TP8/PP2 and DSPARK
parallelism work has since completed (see the optional-profiles section above). Chunked-prefill
sizing, explicit running-request admission control and static memory fraction remain
unswept; if any of them moves the default profile, every delta here will be restated from
the new result files.

<!-- EXTERNAL_COMPARISON_END -->

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

Execute the automated 5-point verification suite to certify node health, virtual API key generation, token-bucket rate limiting, Redis exact-match prompt caching (measured 0.40 ms p50 / 0.82 ms p99 round-trip), and BigQuery trajectory audit streaming:

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
