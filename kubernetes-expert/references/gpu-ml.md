# GPU and ML workloads

## Requesting GPUs

With the NVIDIA device plugin (often installed by the NVIDIA GPU Operator or the cloud
provider), GPUs are an extended resource:
```yaml
resources:
  requests: {cpu: "4", memory: 32Gi}
  limits:
    memory: 32Gi
    nvidia.com/gpu: 1
```
- Put `nvidia.com/gpu` in `limits`; a request defaults to it and, if given, must equal it.
  Whole GPUs only, unless the device plugin is configured for time-slicing or MIG.
- GPU node pools are commonly tainted; add a toleration for the taint your cluster uses
  (`kubectl describe node <gpu-node>` shows it), e.g. `key: nvidia.com/gpu, operator: Exists,
  effect: NoSchedule`.
- Select GPU models with node labels (GPU Feature Discovery publishes labels such as
  `nvidia.com/gpu.product`; confirm with `kubectl get node --show-labels`).
- The image does not need the driver. PyTorch's CUDA wheels bring their CUDA runtime; the
  node supplies the driver (see dockerfile-expert references/python.md).

**Dynamic Resource Allocation** (DRA, stable since 1.34) models devices with DeviceClass and
ResourceClaim objects instead of counted extended resources, so a pod can ask for a GPU by
attributes (memory, model) or share one claim between containers. Use it when the cluster's
GPU driver supports DRA; otherwise the device plugin path above is the default.

## Shared memory for PyTorch DataLoader

PyTorch DataLoader workers pass batches through `/dev/shm`. Container runtimes give a small
default; mount a memory-backed emptyDir:
```yaml
volumes:
  - name: dshm
    emptyDir: {medium: Memory, sizeLimit: 2Gi}
containers:
  - volumeMounts:
      - {name: dshm, mountPath: /dev/shm}
```
Memory-backed emptyDir usage counts against the container's memory limit; size both together.
Verified admitted under `restricted` Pod Security on 1.37.1.

## Model weights

Avoid baking multi-GB weights into the application image (slow pulls, every code release
re-pushes or re-pulls them). Options, in rough order of simplicity:
1. **Image volume** (stable since 1.36): package weights as an OCI image and mount it
   read-only:
   ```yaml
   volumes:
     - name: weights
       image:
         reference: registry.example.com/models/llm-weights:2026-10
         pullPolicy: IfNotPresent
   ```
   Pulled and cached by the node like any image; versioned by tag/digest. Verified admitted
   under `restricted` on 1.37.1 (the Pod Security docs page does not list `image` as an
   allowed volume type; the API server allowed it).
2. **PersistentVolumeClaim** with `ReadOnlyMany` (needs a storage class that supports it, such
   as a network filesystem), populated once by a Job.
3. **Download at startup** into an emptyDir or PVC, gated by a `startupProbe` long enough for
   the download. Pin the model revision (commit hash, not a branch).

## Training and batch jobs

- Use `batch/v1` Job with `restartPolicy: Never`, `podFailurePolicy` (so preemptions and node
  drains do not consume `backoffLimit`), `activeDeadlineSeconds`, and
  `ttlSecondsAfterFinished` (assets/job.yaml).
- Checkpoint to durable storage (PVC or object storage) so a rescheduled pod resumes.
- Multi-node distributed training needs all workers scheduled together. Indexed Jobs
  (`completionMode: Indexed`) give ranks via `JOB_COMPLETION_INDEX`, but coordination and gang
  scheduling usually come from JobSet, Kubeflow Trainer, Kueue, or Volcano. Recent Kubernetes
  releases also add scheduling APIs for pod groups; check the documentation for your cluster
  version before relying on them.
- GPU pods are expensive: set `priorityClassName` so interactive inference can preempt batch
  training, or use Kueue for quotas and queueing.
