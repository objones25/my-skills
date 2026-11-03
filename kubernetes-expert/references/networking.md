# Networking

## Services

- `ClusterIP` (default) for in-cluster traffic. Name ports (`name: http`) and target the
  container's named port (`targetPort: http`).
- `LoadBalancer` provisions a cloud load balancer per Service; fine for one or two L4
  endpoints, expensive as a pattern for many HTTP apps.
- `NodePort` is rarely the right answer outside local clusters.
- Headless (`clusterIP: None`) for StatefulSets and client-side load balancing.
- `internalTrafficPolicy: Local` / `externalTrafficPolicy: Local` keep traffic on the node
  (preserves client IP for external traffic) at the cost of uneven balancing.

## Gateway API vs Ingress

Gateway API (`gateway.networking.k8s.io/v1`: GatewayClass, Gateway, HTTPRoute, GRPCRoute) is
the successor to Ingress. It splits roles: platform owners manage the Gateway (listeners,
TLS, which namespaces may attach), app teams own HTTPRoutes in their namespaces. It supports
header matching, traffic splitting, and cross-namespace references natively, where Ingress
needed controller-specific annotations.

The ingress-nginx controller's maintainers announced retirement in November 2025; per the
Kubernetes blog, best-effort maintenance stopped in March 2026 with no further releases or
security fixes, and existing installs keep running. The blog recommends Gateway API, or
another Ingress controller if you must stay on Ingress.

For new work: write an HTTPRoute (assets/httproute.yaml) attached to the platform's Gateway.
For existing Ingress resources: keep them working, plan the move, and check the conformance
of the chosen Gateway implementation (Envoy Gateway, Istio, Cilium, NGINX Gateway Fabric,
cloud-provider gateways, and others list their status on gateway-api.sigs.k8s.io).

Gateway API CRDs are not part of core Kubernetes; managed clusters may preinstall a version.
Check `kubectl get crd | grep gateway.networking.k8s.io` before applying routes.

## NetworkPolicy patterns

See security.md for default-deny plus DNS. Common allows:
- From the gateway's namespace to the app port (assets/app/networkpolicy.yaml).
- From Prometheus' namespace to the metrics port.
- Egress to a database by namespace and pod labels, or by `ipBlock` for managed databases
  outside the cluster.
Match ports by name where possible so a port renumbering does not silently break policy.
