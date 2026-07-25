# Pinned TensorRT-LLM Blackwell container image for Kimi K3 MXFP4
FROM nvcr.io/nvidia/tensorrt-llm/release:1.2.1

LABEL org.opencontainers.image.title="Kimi K3 MXFP4 Sovereign TensorRT-LLM Serving Runtime"
LABEL org.opencontainers.image.description="Multi-node MPI distributed inference engine for Kimi K3 (2.8T parameters)"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd /root/.ssh \
    && ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa \
    && cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
    && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys \
    && sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config \
    && echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config \
    && echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
    && pip install --no-cache-dir \
       mpi4py \
       requests \
       prometheus-client \
       "huggingface_hub[hf_transfer]>=0.26.0"

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]

