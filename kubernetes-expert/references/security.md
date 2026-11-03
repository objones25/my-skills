# Security

## Pod Security Standards (restricted)

Enforce per namespace with labels (built-in Pod Security Admission, no extra software):
```yaml
metadata:
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```
`enforce` rejects pods; `warn` prints warnings on apply (including for Deployment templates,
before any pod exists); `audit` annotates audit logs. Enforcement applies to Pods only: a
non-compliant Deployment is still created (tested on 1.37.1: dry-run printed
`Warning: would violate PodSecurity "restricted:latest"` and accepted it), and then its
ReplicaSet fails to create pods. If a rollout shows 0 pods, check
`kubectl describe rs` / `kubectl get events` for `FailedCreate ... violates PodSecurity`. Pin
`pod-security.kubernetes.io/enforce-version: v1.37` if you want upgrades not to change the
policy under you.

What `restricted` requires (from the Pod Security Standards page; each verified against a
1.37.1 API server, see scripts/tests/pss_cases):
- `runAsNonRoot: true` at pod or container level; `runAsUser` never 0
- `allowPrivilegeEscalation: false` on every container
- `capabilities.drop: ["ALL"]`; `add` may contain only `NET_BIND_SERVICE`
- `seccompProfile.type` `RuntimeDefault` or `Localhost` at pod or container level
- no `privileged`, `hostNetwork`, `hostPID`, `hostIPC`, `hostPort`, `hostPath`
- volumes limited to configMap, csi, downwardAPI, emptyDir, ephemeral, persistentVolumeClaim,
  projected, secret. Observed discrepancy: the 1.37.1 API server also admitted an `image`
  volume under restricted, though the docs page (checked 2026-10-19) does not list it.

Not required by `restricted` but recommended: `readOnlyRootFilesystem: true`, which is why the
checker reports it separately (K010).

The image and the manifest must agree: an image with `USER app` (a name) plus
`runAsNonRoot: true` and no `runAsUser` fails at container creation with "image has
non-numeric user". Use numeric `USER` in the Dockerfile and matching `runAsUser`.

**User namespaces** (stable since 1.36): `hostUsers: false` maps container root to an
unprivileged host UID. Needs node and runtime support (recent kernel, containerd or CRI-O
versions); a strong extra layer for workloads that must run as UID 0 inside.

## Service accounts and RBAC

- One ServiceAccount per app; `automountServiceAccountToken: false` on the SA and pod unless
  the app calls the Kubernetes API.
- If it does: a namespaced `Role` with exact resources and verbs, bound by `RoleBinding`.
  Avoid `ClusterRole` unless it truly needs cluster scope; never bind `cluster-admin` to an
  app. Watch for privilege-escalating verbs: `create pods`, `get secrets`, `escalate`, `bind`,
  `impersonate`.
- Cloud credentials via workload identity (EKS Pod Identity or IRSA, GKE Workload Identity,
  AKS Workload Identity), not static keys in Secrets.

## Secrets

- Kubernetes Secrets are base64, not encrypted. Enable encryption at rest (managed clusters
  often do by default; check), restrict `get/list secrets` in RBAC.
- Keep them out of git in plain form: External Secrets Operator (syncs from a cloud secret
  manager), Secrets Store CSI Driver (mounts directly), SOPS (encrypted files decrypted by
  the deploy tool), or Sealed Secrets (cluster-side decryption).
- Prefer mounting Secrets as files over env vars when the app supports it: env vars leak into
  crash dumps, child processes, and `kubectl describe` output of some tools.
- Never put a secret in a ConfigMap, an annotation, a command-line argument, or a literal
  `env.value` (checker rule K017).

## NetworkPolicy

Needs a CNI that enforces it (Calico, Cilium, and most managed-cluster network policy
options; plain flannel does not). Policies are additive allow-lists once any policy selects a
pod.

Default deny ingress and egress, then allow DNS:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
spec:
  podSelector: {}
  policyTypes: ["Ingress", "Egress"]
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns
spec:
  podSelector: {}
  policyTypes: ["Egress"]
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: kube-system
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - {port: 53, protocol: UDP}
        - {port: 53, protocol: TCP}
```
Then add explicit egress for each dependency (database CIDR or namespace, external APIs).
Default-deny egress breaks things that were silently working; roll it out with the app
owner. The `kube-dns` label is the common default; confirm with
`kubectl -n kube-system get pods --show-labels`.

## Images

- Private registry: `imagePullSecrets` on the ServiceAccount (applies to all its pods) rather
  than every pod spec.
- Pin digests from CI. Admission policies (Kyverno, OPA Gatekeeper, or built-in
  ValidatingAdmissionPolicy with CEL) can require digests, trusted registries, and signatures.
