# 1. Base Image: Use the official NVIDIA PyTorch container.
# It includes Python, CUDA, cuDNN, and all necessary bindings.
# Provided by Ilya Kryukov, NVIDIA employee.
# Does not included in CI, because it is too large to download (~15GB)
FROM nvcr.io/nvidia/pytorch:25.06-py3
# ================================================ #

# 2. Metadata
LABEL maintainer="Reisen Raumberg (attn-signs) <fallturm.bremen@gmail.com>"
LABEL version="0.2.0"
LABEL description="MyLLM is a high-performance framework for fine-tuning LLMs, optimized for Docker and Kubernetes."
# ================================================ #

# 3. Environment Variables
# Set non-interactive frontend for package installations to prevent prompts.
ENV DEBIAN_FRONTEND=noninteractive
# Set Python to run in unbuffered mode, which is recommended for containers.
ENV PYTHONUNBUFFERED=1
# Set paths for Hugging Face cache to be in a predictable location.
ENV HF_HOME="/root/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/root/.cache/huggingface/models"
# Set paths for WandB
ENV WANDB_DIR="/libllm/wandb"
ENV WANDB_CACHE_DIR="/root/.cache/wandb"
# ================================================ #

# 4. System Dependencies
# Install essential tools. git is needed to install some pip packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    pdsh \
    openssh-server \
    && rm -rf /var/lib/apt/lists/* 
# ================================================ #

# 5. Application Setup
WORKDIR /libllm
# ================================================ #

# 6. Install Project Dependencies
# Copy only the necessary files first to leverage Docker's layer caching.
# This prevents re-installing all dependencies on every source code change.
COPY pyproject.toml README.md requirements-dev.txt ./
# The base image already contains torch, so we can use --no-deps for it
# to avoid conflicts and speed up the installation.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements-dev.txt
# ================================================ #

# 7. Copy Source Code
# Now copy the rest of the application code.
COPY . .
# ================================================ #

# 7. Copying the requirements file:
RUN pip install --no-cache-dir -e '.[kubernetes]'
# Force reinstall bitsandbytes to ensure it's compiled against the container's CUDA version
RUN pip uninstall -y bitsandbytes && pip install --no-cache-dir bitsandbytes
# ================================================ #

# 8. Create cache directories
# This ensures the directories exist when the container starts.
RUN mkdir -p $HF_HOME $TRANSFORMERS_CACHE $WANDB_DIR $WANDB_CACHE_DIR
# ================================================ #

# 9. Final Touches
# The container is now ready. It does not need an ENTRYPOINT or CMD,
# as the command will be provided by the Kubernetes TrainJob manifest.
# This makes the container more flexible and follows best practices.
CMD []