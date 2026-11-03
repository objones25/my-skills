---
name: kubernetes-expert
description: Write, review, and fix Kubernetes manifests (Deployments, StatefulSets, Jobs/CronJobs, Services, Gateway API routes, PodDisruptionBudgets, HPAs, NetworkPolicies, ServiceAccounts, kustomize overlays, Helm output) to production best practices - restricted Pod Security, resource requests and limits, correct probes, graceful shutdown, disruption budgets, topology spread, and safe secret handling. Ships a kustomize base that passes server-side validation and the "restricted" Pod Security Standard on Kubernetes 1.37, a batch/GPU Job template, and a checker whose Pod Security verdicts match the real API server. Use this whenever the user asks to deploy something to Kubernetes, write or review YAML for k8s/EKS/GKE/AKS/k3s/kind, debug CrashLoopBackOff, OOMKilled, pending pods or failed rollouts, set resources or probes, or migrate from Ingress to Gateway API, even if they just say "deploy this container". For Dockerfiles use dockerfile-expert.
---

# Kubernetes expert

Goal: manifests that pass `restricted` Pod Security, schedule predictably (requests set),
survive node drains and rollouts without dropping traffic (probes, PDB, spread, graceful
shutdown), grant no more access than needed, and contain no secrets in plain text. Prove it
by running validators, not by reading YAML.

## Workflow

1. **Gather facts first**: the container image and its USER (must be numeric; see
   dockerfile-expert), listening port, health endpoints, whether it writes to disk, CPU/memory
   observed under load (or say requests are estimates), stateful or not, how it gets config
   and secrets, how traffic arrives (Gateway API, Ingress controller, LoadBalancer), the
   cluster version (`kubectl version`), and the delivery tool (plain kubectl, kustomize, Helm,
   Argo CD/Flux).
2. **Pick the workload kind** (references/workloads.md): Deployment for stateless services,
   StatefulSet only for stable identity or per-pod storage, Job/CronJob for finite work,
   DaemonSet for per-node agents.
3. **Start from `assets/app/`** (kustomize base: Namespace with Pod Security labels,
   ServiceAccount, Deployment, Service, PDB, HPA, NetworkPolicy) or `assets/job.yaml`, plus
   `assets/httproute.yaml` for Gateway API. Replace `myapp`, image, port, UID, and resource
   numbers. Keep every security field; remove a feature only with a stated reason.
4. **Validate**, showing real output for whatever is available:
   ```
   kubectl kustomize . | python3 scripts/check_manifests.py -
   kubeconform -strict -summary -kubernetes-version 1.37.0 .
   kubectl apply --dry-run=server -k .
   kubectl diff -k .
   ```
   Server-side dry-run is the strongest check: it runs real admission, including Pod
   Security. Without cluster access, run the checker and kubeconform and say so.
5. **Deliver** the manifests plus a short note: what the resource numbers are based on, which
   endpoints the probes hit and why, what the NetworkPolicy allows, where secrets come from,
   and what is unverified.

When reviewing existing manifests or a Helm chart, render them first (`helm template`,
`kubectl kustomize`), run the checker, then walk `references/review-checklist.md`. Report by
impact: security, then availability, then scheduling and cost, then hygiene.

## Rules and why they matter

**Security (every pod)**
- Namespace labels `pod-security.kubernetes.io/enforce: restricted` (plus `audit` and `warn`).
  Tested on a 1.37.1 API server (2026-10-19, machine not recorded): a pod with no securityContext is rejected naming four
  fields; the template's pod is admitted.
- Pod `securityContext`: `runAsNonRoot: true`, numeric `runAsUser`/`runAsGroup` matching the
  image, `seccompProfile.type: RuntimeDefault`. Container `securityContext`:
  `allowPrivilegeEscalation: false`, `capabilities.drop: ["ALL"]`,
  `readOnlyRootFilesystem: true` with `emptyDir` mounts for the paths the app writes.
- Dedicated ServiceAccount, `automountServiceAccountToken: false` unless the app calls the
  API; then a namespaced Role with only the verbs it uses.
- Secrets via `secretKeyRef` or a mounted Secret, never a literal `value:`. Kubernetes
  Secrets are only base64-encoded; keep them out of git (External Secrets Operator, Secrets
  Store CSI driver, SOPS, or Sealed Secrets) and turn on encryption at rest.
- Default-deny NetworkPolicy per namespace, then allow specific traffic (needs a CNI that
  enforces it).

**Scheduling and resources**
- Every container sets CPU and memory requests (the scheduler places by requests; the HPA
  computes utilization against them). Set a memory limit, usually equal to the request.
- CPU limits are a tradeoff, not a default: they cause CFS throttling and latency spikes even
  on idle nodes; leaving them off lets pods burst. Set them when tenants must be isolated or
  for Guaranteed QoS. State the choice. Details in references/resources-probes.md.
- Set the runtime's own memory ceiling below the container limit (`GOMEMLIMIT`,
  `--max-old-space-size`, JVM `-XX:MaxRAMPercentage`) so it garbage-collects before the
  kernel OOM-kills it.

**Availability**
- Probes: `readinessProbe` gates traffic and may check what the pod needs to serve;
  `livenessProbe` restarts the container and must check only the process (a liveness check
  on a shared database restarts every pod during a database blip); `startupProbe` covers
  slow starts so liveness can stay strict.
- At least 2 replicas (or HPA `minReplicas: 2`), a `policy/v1` PDB, and
  `topologySpreadConstraints` across hostname and zone.
- Rollouts: `maxUnavailable: 0`, `maxSurge: 1` for small services.
- Graceful shutdown: the app handles SIGTERM, a `preStop` `sleep` (stable since 1.34, needs no
  shell) covers endpoint-removal lag, and `terminationGracePeriodSeconds` exceeds preStop plus
  shutdown time.
- With an HPA, omit `spec.replicas` from the Deployment you apply, or every apply resets the
  scale.

**Hygiene**
- Images pinned by tag plus digest from CI; never `:latest`.
- `app.kubernetes.io/name`, `component`, `part-of` labels; selectors use a stable subset
  (Deployment selectors are immutable after creation).
- Never deploy into `default`. One namespace per app or team boundary.
- New HTTP routing uses Gateway API (`HTTPRoute`). The ingress-nginx project's best-effort
  maintenance ended in March 2026 per the Kubernetes blog; see references/networking.md.

## Files

- `assets/app/`: kustomize base (namespace, serviceaccount, deployment, service, pdb, hpa,
  networkpolicy). Passed `kubectl apply --dry-run=server` on Kubernetes 1.37.1 with Pod
  Security `restricted` enforced, and kubeconform for 1.37.0.
- `assets/job.yaml`: batch Job with podFailurePolicy, deadlines, TTL, and a commented GPU
  request; `assets/httproute.yaml`: Gateway API route (validated against the CRD schema)
- `references/workloads.md`: choosing kinds, rollouts, config changes, StatefulSets, Jobs
- `references/resources-probes.md`: requests/limits, QoS, CPU limits debate, runtime memory
  settings, probe design, in-place resize
- `references/security.md`: Pod Security fields, RBAC, secrets, NetworkPolicy with DNS egress,
  image pull secrets, user namespaces
- `references/networking.md`: Services, Gateway API vs Ingress, NetworkPolicy patterns
- `references/gpu-ml.md`: GPU requests, tolerations, /dev/shm for PyTorch, DRA, image volumes
  for model weights, Indexed Jobs for distributed work
- `references/review-checklist.md`: ordered review list
- `scripts/check_manifests.py`: policy checker (`--rules`, `--json`, `--strict`, reads files,
  directories, or stdin). Needs PyYAML. Tests: `python3 scripts/tests/test_check_manifests.py`

Versions were current on 19 October 2026 (Kubernetes 1.37, with 1.36 and 1.35 also
supported). Check `kubectl version` against the cluster and adjust.
