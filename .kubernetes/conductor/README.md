# Conductor: Installing the Kubeflow Trainer Operator

This directory contains the patched Kubernetes manifests required to install the **Kubeflow Trainer** operator. The operator acts as a custom controller in your cluster, responsible for managing the lifecycle of `TrainJob` resources.

## What Will Be Installed?

Applying these manifests will create the following resources in your cluster:

*   **`kubeflow-system` Namespace**: A dedicated namespace to house all the operator's components.
*   **Custom Resource Definitions (CRDs)**: Defines the `TrainJob`, `TrainingRuntime`, and other custom resources that the operator manages.
*   **Controller Manager Deployments**: Two deployments (`jobset-controller-manager` and `kubeflow-trainer-controller-manager`) that contain the core logic of the operator.
*   **RBAC Rules**: The necessary `ClusterRoles` and `ClusterRoleBindings` to grant the operator permissions to manage pods, services, and other resources across the cluster.
*   **Services and Webhooks**: Internal services for metrics and webhook validation.

## Installation Command

To install the operator, execute the following `kubectl` command from the parent `.kubernetes` directory:

```bash
# This command recursively finds and applies all kustomization.yaml files.
kubectl apply --server-side -k conductor/
```

The `--server-side` flag is recommended for applying CRDs and operator manifests, as it helps prevent conflicts if you need to re-apply the configuration. The `-k` flag tells `kubectl` to use `kustomize` to build the final manifests from the directory.

## Verifying the Installation

After running the command, allow a minute for all the components to be downloaded and started. You can monitor the progress with the following commands.

**1. Check the Pods:**

Ensure that the controller manager pods are running correctly in the `kubeflow-system` namespace.

```bash
kubectl get pods -n kubeflow-system
```

You should see two pods, both with a `STATUS` of `Running` and `READY` counts of `1/1`.

```
NAME                                                   READY   STATUS    RESTARTS   AGE
jobset-controller-manager-78bcbf6455-5c67m             1/1     Running   0          ...
kubeflow-trainer-controller-manager-678d4bfc86-8mf8q   1/1     Running   0          ...
```

**2. Check the Training Runtimes:**

Verify that the predefined "recipes" for different distributed training frameworks have been successfully registered.

```bash
kubectl get clustertrainingruntimes
```

You should see a list of available runtimes, such as `deepspeed-distributed`, `torch-distributed`, etc.

```
NAME                    AGE
deepspeed-distributed   ...
mlx-distributed         ...
mpi-distributed         ...
torch-distributed       ...
...
```

If both commands execute successfully and show the expected resources, the Conductor has been installed correctly and is ready to manage your training jobs. 