# Kubernetes Configuration for Distributed LLM Training

This directory contains all the necessary Kubernetes manifests to deploy and operate **Kubeflow Trainer**, a powerful operator for orchestrating distributed training jobs. The configuration is split into two logical parts:

1.  **`conductor`**: Contains the core manifests for installing the Kubeflow Trainer operator itself. Think of this as installing the "conductor" who will direct our training jobs.
2.  **`orchestra`**: Contains the example `TrainJob` manifest for running a distributed training workload. This is the "orchestra" that the conductor will lead.

Manifests for Trainer were carefully overriden to match k3s from:
https://www.kubeflow.org/docs/components/trainer/
https://github.com/kubeflow/trainer

## Deployment Workflow

The deployment process is designed to be sequential and straightforward:

1.  **Set up a Kubernetes Cluster**: Before you begin, ensure you have a running Kubernetes cluster with GPU support. For a detailed guide on setting up a `k3s` cluster from scratch, please refer to the "Distributed Training with Kubernetes" section in the main project `README.md`.
2.  **Deploy the Conductor**: Install the Kubeflow Trainer operator using the manifests in the `conductor` directory. This is a one-time setup step. Follow the instructions in `conductor/README.md`.
3.  **Run the Orchestra**: Once the operator is running, you can submit your training jobs using the `TrainJob` manifest in the `orchestra` directory. Follow the instructions in `orchestra/README.md`.

## ⚠️ Important Note on `k3s` Compatibility

The manifests provided here have been **specifically patched to ensure compatibility with `k3s`**. A standard Kubeflow Trainer installation often fails on `k3s` due to strict webhook validation policies and networking quirks.

The key patch is located in `conductor/webhook/kustomization.yaml`, which forces the `failurePolicy` for validating webhooks to `Ignore`. This prevents the installation from stalling while waiting for network routes to be established.

**Do not attempt to apply the official manifests directly from the Kubeflow GitHub repository, as this will likely fail in a `k3s` environment.** Always use the patched versions contained within this directory. 

When everything is ready, make sure to apply custom attn-signs manifest (kubectl apply -f .kubernetes/orchestra/kubernetes-manifest.yaml)

---
# Step-by-step configuraion
---

This project includes a pre-configured, patched setup for running distributed training jobs on a Kubernetes cluster using **Kubeflow Trainer**. The provided manifests in the `.kubernetes` directory are specifically tailored for `k3s` to work around common networking issues.

Follow these steps to set up a GPU-enabled k3s cluster and deploy the training operator.

### 1. Setting up the k3s Cluster

We recommend `k3s` for a lightweight, easy-to-manage Kubernetes distribution.

**On the Master Node:**

```bash
# Install k3s. We disable the default traefik ingress as we don't need it.
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik --docker" sh -

# Verify the master node is ready
sudo k3s kubectl get node
```

**On each Worker Node:**

First, get the join token from the master node:
```bash
# On the master node
sudo cat /var/lib/rancher/k3s/server/node-token
```

Then, use the token and the master's IP to join the worker to the cluster:
```bash
# On the worker node
curl -sfL https://get.k3s.io | K3S_URL=https://<MASTER_IP>:6443 K3S_TOKEN=<YOUR_TOKEN> INSTALL_K3S_EXEC="--docker" sh -
```

**Configure `kubectl` on your local machine:**

Copy the config file from the master node to your local `~/.kube/config` and replace the `127.0.0.1` address with your master node's IP.

```bash
# On your local machine
scp user@<MASTER_IP>:/etc/rancher/k3s/k3s.yaml ~/.kube/config
sed -i 's/127.0.0.1/<MASTER_IP>/g' ~/.kube/config
kubectl get nodes
```

### 2. Enabling NVIDIA GPU Support

This step is crucial for the cluster to recognize and schedule workloads on your GPUs.

**On ALL GPU-enabled nodes (master and workers):**
1.  Ensure you have the official NVIDIA drivers installed.
2.  Install the NVIDIA container toolkit.

**On your local machine (with `kubectl`):**

Deploy the NVIDIA Kubernetes Device Plugin. This allows the Kubernetes scheduler to see the GPUs.
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

Verify that the GPUs are now visible as a resource in your nodes:
```bash
# Check each node that has GPUs
kubectl describe node <your-gpu-node-name> | grep nvidia.com/gpu
```
You should see a line like `nvidia.com/gpu: 2` indicating the number of available GPUs.

### 3. Deploying Kubeflow Trainer

Now, we deploy our patched version of Kubeflow Trainer. The manifests are located in the `.kubernetes` directory. For a detailed explanation, please see the README files inside that directory.

To deploy the operator (the "Conductor"), run the following command from the project root:
```bash
kubectl apply --server-side -k .kubernetes/conductor/
```
This will install all the necessary CRDs, services, and controllers into the `kubeflow-system` namespace.

### 4. Running a Training Job

Once the operator is running, you can submit a training job (the "Orchestra").

First, configure your job by editing `.kubernetes/orchestra/kubernetes-manifest.yaml`. You **must** specify your Docker image and training command.

Then, launch the job:
```bash
kubectl apply -f .kubernetes/orchestra/kubernetes-manifest.yaml
```

For detailed instructions on how to monitor the job, see `.kubernetes/orchestra/README.md`.