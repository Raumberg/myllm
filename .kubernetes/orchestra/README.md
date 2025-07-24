# Orchestra: Running a Distributed Training Job

This directory contains an example `TrainJob` manifest (`kubernetes-manifest.yaml`) for launching a distributed training run using our `myllm` framework.

## The `TrainJob` Resource

A `TrainJob` is a custom resource (CRD) provided by Kubeflow Trainer. It's a high-level abstraction that simplifies the process of launching distributed jobs. Key fields in our example include:

*   **`spec.trainingRuntime`**: Specifies which "recipe" to use. We use `deepspeed-distributed` for our example.
*   **`spec.replicas`**: The total number of pods to create for the job. For a 2-node, 4-GPU setup, this will be `2`.
*   **`spec.replicaSpecs`**: This is where we define the different roles for our job.
    *   **`launcher`**: The main pod that initiates the `deepspeed` command. There is always exactly one launcher.
    *   **`worker`**: The other pods that participate in the training. The number of workers is `spec.replicas - 1`.
*   **`resources.limits`**: This is where we request the necessary number of GPUs (`nvidia.com/gpu`) for each pod.
*   **`affinity`**: We use a `podAntiAffinity` rule to ensure that Kubernetes schedules the launcher and worker pods on different physical nodes, maximizing resource utilization.

## 1. Configuration

Before launching the job, you **must** configure the `kubernetes-manifest.yaml` file:

1.  **`image`**: In both the `launcher` and `worker` sections, change the `image` field from the placeholder (`your-registry/myllm-framework:latest`) to the full path of your actual `myllm` Docker image. Ensure this image is accessible to your Kubernetes cluster (i.e., pushed to a registry).
2.  **`command`**: In the `launcher` section, review and update the training `command`. Pay close attention to the `deepspeed` arguments (`--num_gpus`) and the path to your training configuration file (`--config`).

## 2. Launching the Job

Once configured, apply the manifest to submit the job to the cluster. Run this command from the parent `.kubernetes` directory:

```bash
kubectl apply -f orchestra/kubernetes-manifest.yaml
```

This will create a `TrainJob` resource, which the Kubeflow Trainer operator will then see and use to create the necessary pods.

## 3. Monitoring the Job

Monitoring a distributed job involves checking the state of the `TrainJob` itself and the pods it creates.

**1. Check the `TrainJob` Status:**
This gives you a high-level overview of the job's state (e.g., `Created`, `Running`, `Succeeded`, `Failed`).

```bash
# Replace the job name and namespace if you changed them
kubectl get trainjob myllm-sft-h100-distributed-job -n default -o yaml
```
Look at the `status.conditions` field in the YAML output for detailed information.

**2. List the Training Pods:**
This command finds all pods associated with the training job.

```bash
kubectl get pods -n default -l app.kubernetes.io/name=trainjob
```
You should see pods for the launcher and worker(s), eventually reaching a `Running` state.

**3. Stream Logs from the Launcher:**
The most important logs (containing training progress, metrics, etc.) will come from the `launcher` pod.

First, find the exact name of the launcher pod:
```bash
# The launcher pod name usually ends in "-launcher-0"
kubectl get pods -n default | grep launcher
```

Then, stream its logs:
```bash
# Replace with the actual pod name from the command above
kubectl logs -f <your-launcher-pod-name> -n default
```
This will give you a real-time view of your training process. 