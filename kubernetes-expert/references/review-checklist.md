# Reviewing Kubernetes manifests

Render first if templated (`helm template . -f values.yaml`, `kubectl kustomize overlays/x`),
then run and quote:
```
... | python3 scripts/check_manifests.py -
kubeconform -strict -summary -kubernetes-version <cluster version> .
kubectl apply --dry-run=server -f rendered.yaml
```
Report in this order, each with the exact YAML change.

## 1. Security
- [ ] Namespace enforces `restricted` Pod Security (or a documented exception)
- [ ] Pod and container securityContext meet restricted; `readOnlyRootFilesystem: true`
- [ ] `runAsUser` matches the image's numeric USER
- [ ] Dedicated ServiceAccount; token not mounted unless needed; RBAC least privilege
- [ ] No literal secrets in env, args, ConfigMaps, or annotations; secret source documented
- [ ] NetworkPolicy default-deny with explicit allows
- [ ] Images from trusted registries, pinned by digest in what actually deploys

## 2. Availability
- [ ] readinessProbe on every serving container; liveness checks only the process;
      startupProbe for slow starts
- [ ] Graceful shutdown: SIGTERM handled, preStop sleep, grace period long enough
- [ ] 2+ replicas (or HPA min 2), PDB, topology spread across nodes and zones
- [ ] Rollout strategy keeps capacity (`maxUnavailable: 0`), `rollout status` gated in CI
- [ ] Selectors match template labels; Service selectors match pods; probe ports exist
- [ ] HPA-managed Deployments do not set `spec.replicas` in applied manifests

## 3. Scheduling and cost
- [ ] CPU and memory requests from observed usage; memory limits set
- [ ] CPU limit choice stated (and why)
- [ ] Runtime memory ceiling below the container limit (GOMEMLIMIT, max-old-space-size, JVM)
- [ ] GPU workloads: tolerations, node selection, /dev/shm, weights not baked into the image

## 4. Hygiene
- [ ] No removed or deprecated apiVersions (checker K015; `kubectl convert` or Pluto for more)
- [ ] Not in `default` namespace; recommended `app.kubernetes.io/*` labels
- [ ] Config changes trigger rollouts (hash-suffixed generators or checksum annotations)
- [ ] Gateway API for new HTTP routing
