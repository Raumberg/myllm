FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

LABEL maintainer="Reisen Raumberg (attn-signs) <fallturm.bremen@gmail.com>"
LABEL version="0.1.2"
LABEL description="MyLLM is a LLM framework for training and fine-tuning LLMs."

# Set non-interactive frontend for package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, Python 3.12, and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN uv python install 3.12.8
RUN uv venv /opt/venv --python python3.12
ENV PATH="/opt/venv/bin:$PATH"

# Set up the working directory
WORKDIR /libllm

# Copy only the files needed for dependency installation to leverage caching
COPY . .

# Install all dependencies using uv.
RUN uv pip install -e .

# Set environment variables for Hugging Face and others
ENV PATH="/opt/venv/bin:$PATH"
ENV HF_HOME="/libllm/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/libllm/.cache/huggingface/models"
ENV WANDB_DIR="/libllm/wandb"
ENV WANDB_CACHE_DIR="/libllm/.cache/wandb"

# Prepare cache directories
RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $WANDB_DIR $WANDB_CACHE_DIR && \
    chmod -R a+rx /opt/venv/bin /root/.local/share/uv

ENTRYPOINT ["/bin/bash", "-i", "-c", "source /opt/venv/bin/activate && exec bash -i"]
CMD []