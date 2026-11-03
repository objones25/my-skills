# Resources and probes

## Requests and limits

- **Requests** are what the scheduler reserves. A pod with no requests can be placed on a full
  node and is first in line for eviction. The HPA's CPU utilization is measured against
  requests, so an HPA without requests does nothing useful.
- **Memory limit**: exceeding it gets the container OOM-killed (`OOMKilled`, exit 137). Set
  it, usually equal to the request: memory is not compressible, and limit > request
  overcommits the node.
- **CPU limit**: exceeding it throttles (CFS quota), even when the node has idle CPU.
  Arguments against: latency spikes for bursty request handlers and multi-threaded runtimes
  that use a full quota in a few milliseconds. Arguments for: noisy-neighbor isolation in
  shared clusters, predictable cost, and Guaranteed QoS (requests == limits for both CPU and
  memory) which some node-level CPU pinning needs. Pick per workload and write down why.
- **Sizing**: start from observed usage (`kubectl top pod`, Prometheus
  `container_memory_working_set_bytes`, `container_cpu_usage_seconds_total`), request around
  the p90 of normal load, set memory limit with headroom over peak. If there is no data, say
  the numbers are guesses and suggest VPA in recommendation mode (`updateMode: "Off"`).
- **QoS classes**: Guaranteed (requests == limits for every container), Burstable (some
  requests), BestEffort (none). Under node memory pressure BestEffort is evicted first.
- **In-place resize** (stable since 1.35): CPU and memory requests/limits can change on a
  running pod via the `resize` subresource; per-container `resizePolicy` decides whether a
  restart is needed. VPA can use this instead of recreating pods.
- **Pod-level resources** (beta, on by default since 1.34): `spec.resources` on the pod sets a
  shared budget for all its containers.

## Make the runtime respect the limit

The kernel kills on the cgroup limit; runtimes size their heaps from other signals.
- Go: `GOMEMLIMIT` at ~90% of the limit. With the downward API:
  ```yaml
  env:
    - name: GOMEMLIMIT
      valueFrom:
        resourceFieldRef: {resource: limits.memory, divisor: "1"}
  ```
  (that sets 100%; prefer a literal like `460MiB` for a 512Mi limit).
- Node: `NODE_OPTIONS=--max-old-space-size=<MB>` below the limit (heap only; buffers and
  native memory come on top).
- JVM: `-XX:MaxRAMPercentage=75`.
- Python: no heap knob; size worker counts (`--workers`) so that workers x per-worker memory
  fits, and remember forked workers share memory only until pages are written.

Thread pools: the container sees the node's CPU count. Set worker/thread counts explicitly
(or `GOMAXPROCS` from `limits.cpu`) rather than trusting auto-detection.

## Probes

| Probe | Failing means | Should check | Must not check |
|---|---|---|---|
| startupProbe | Not started yet; liveness and readiness are paused until it passes | Process is up and initialized | Anything that can take longer than `failureThreshold x periodSeconds` |
| readinessProbe | Remove from Service endpoints (no traffic), keep running | Can this pod serve right now: warmed caches, required downstreams reachable | Heavy work; it runs every few seconds per pod |
| livenessProbe | Kill and restart the container | The process itself responds (event loop not wedged, not deadlocked) | Databases, other services, disk space: shared failures then restart every pod at once |

Rules of thumb:
- Separate endpoints: `/healthz` (liveness, trivial) and `/readyz` (readiness, can check
  dependencies). Identical liveness and readiness checks are a smell the checker flags (K008).
- No liveness probe at all is better than a wrong one. Add liveness to catch hangs, not
  crashes (crashes already restart the container).
- Use `startupProbe` for slow boots (model loading) instead of a large
  `initialDelaySeconds` on liveness.
- Probe the named port (`port: http`) so a port change cannot desync.
- gRPC services can use the built-in `grpc` probe type with the standard health service.
- Probe timeouts default to 1 s; a GC pause or cold cache can exceed it. Set `timeoutSeconds`
  deliberately for heavy runtimes.

## Debugging quick reference

```
kubectl describe pod POD
kubectl get events --sort-by=.lastTimestamp -n NS
kubectl logs POD -c CONTAINER --previous
kubectl debug -it POD --image=busybox:1.37 --target=CONTAINER
kubectl top pod -n NS --containers
```
- `CrashLoopBackOff`: read `--previous` logs; check exit code in describe (137 = SIGKILL,
  often OOM; 143 = SIGTERM; 1 = app error).
- `Pending`: describe shows the scheduler's reason (insufficient cpu/memory, taints, PVC not
  bound, topology constraints).
- `CreateContainerConfigError`: missing ConfigMap/Secret key, or `runAsNonRoot` with a
  non-numeric image USER ("image has non-numeric user ... cannot verify user is non-root").
- `ImagePullBackOff`: wrong name/tag, missing `imagePullSecrets`, or architecture mismatch.
- `kubectl debug` with `--target` shares the process namespace, which is how to inspect a
  distroless container that has no shell.
