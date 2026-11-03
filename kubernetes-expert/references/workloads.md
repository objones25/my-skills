# Workload kinds, rollouts, and configuration

## Choosing a kind

| Kind | Use for | Not for |
|---|---|---|
| Deployment | Stateless services, workers reading a queue | Anything needing a stable network identity or its own disk |
| StatefulSet | Databases, brokers, anything needing stable names (`pod-0`) and per-pod PVCs | Stateless apps "because they have a volume"; a shared PVC works with a Deployment if the storage supports it |
| Job | Finite work: migrations, batch processing, training runs | Long-running services |
| CronJob | Scheduled Jobs | Sub-minute schedules |
| DaemonSet | One pod per node: log shippers, node agents, CNI | Application services |

Prefer a managed service over a StatefulSet for production databases unless the team operates
that database already.

## Rollouts

- `strategy.rollingUpdate`: `maxUnavailable: 0`, `maxSurge: 1` (or 25%) means capacity never
  drops. Requires spare cluster capacity for the surge pod.
- `minReadySeconds: 10` makes a pod count as available only after staying ready that long,
  which catches crash-after-start.
- `progressDeadlineSeconds` (default 600) marks a stuck rollout as failed;
  `kubectl rollout status deploy/myapp --timeout=5m` in CI fails the pipeline on it.
- `revisionHistoryLimit: 5` keeps `kubectl rollout undo` possible without piling up ReplicaSets.
- Deployment `spec.selector` is immutable. Choose labels once (`app.kubernetes.io/name` plus
  `component`) and never put version labels in the selector.

## Graceful shutdown sequence

On deletion, the kubelet runs `preStop`, then sends SIGTERM, then SIGKILL after
`terminationGracePeriodSeconds` (counted from the start, including preStop). Endpoint removal
happens in parallel, so load balancers may still send requests for a few seconds. Hence:
1. `preStop: sleep: {seconds: 5}` keeps the old pod serving while endpoints update (the sleep
   action is stable since 1.34 and needs no shell, so it works on distroless).
2. The app handles SIGTERM: stop accepting, finish in-flight requests, exit.
3. `terminationGracePeriodSeconds` > preStop + drain time.

## Configuration changes

A pod does not restart when a ConfigMap or Secret it uses as env changes. Options:
- kustomize `configMapGenerator`/`secretGenerator` append a content hash to the name, so a
  change produces a new name and therefore a rollout. Preferred with kustomize.
- Helm: a `checksum/config` annotation on the pod template computed from the rendered config.
- Mounted (not env) ConfigMaps update in place eventually; only useful if the app reloads.
- `immutable: true` on ConfigMaps and Secrets that never change reduces API server load.

## StatefulSets

- `serviceName` must point at a headless Service (`clusterIP: None`).
- `volumeClaimTemplates` creates one PVC per pod; PVCs survive scale-down and StatefulSet
  deletion unless `persistentVolumeClaimRetentionPolicy` says otherwise.
- `podManagementPolicy: Parallel` if pods do not need ordered startup.
- Updates: `updateStrategy.rollingUpdate.partition` for canaries.

## Jobs and CronJobs

- `restartPolicy: Never` with `backoffLimit` gives a fresh pod per retry (logs preserved);
  `OnFailure` restarts in place.
- `activeDeadlineSeconds` bounds wall time; `ttlSecondsAfterFinished` cleans up.
- `podFailurePolicy` (requires `restartPolicy: Never`) distinguishes permanent failures
  (exit codes) from disruptions (`DisruptionTarget` condition) so evictions do not burn retries.
  The template in assets/job.yaml passed server-side dry-run on 1.37.1.
- `completionMode: Indexed` gives each pod `JOB_COMPLETION_INDEX` for sharded work.
- CronJob: `concurrencyPolicy: Forbid` unless overlap is safe, `startingDeadlineSeconds`,
  `timeZone: "America/Los_Angeles"` instead of UTC arithmetic, and keep
  `successfulJobsHistoryLimit` small.
- Database migrations run as a Job (or Helm hook / Argo CD sync hook) before the new
  Deployment, never in every replica's startup command.

## Sidecars

Native sidecars (stable since 1.33) are init containers with `restartPolicy: Always`: they
start before app containers, keep running, and stop after them. Use this instead of plain
extra containers for proxies and log shippers, especially in Jobs (a plain sidecar keeps the
Job from completing).
