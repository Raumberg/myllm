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