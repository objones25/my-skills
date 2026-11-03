#!/usr/bin/env python3
"""Best-practice checker for Kubernetes manifests (policy, not schema).

Schema validity is kubeconform's or `kubectl apply --dry-run=server`'s job. This script
checks what valid manifests still get wrong: missing requests/limits/probes, pods that
the "restricted" Pod Security Standard would reject, selectors that match nothing,
replicated workloads without a PodDisruptionBudget, removed API versions, and secrets
inlined as plain env values.

Usage:
    check_manifests.py [--json] [--strict] [--min-severity LEVEL] [--ignore K002,...] PATH...
    kubectl kustomize overlays/prod | check_manifests.py -
    helm template . | check_manifests.py -
    check_manifests.py --rules

PATH may be a file, a directory (searched for *.yaml / *.yml), or '-' for stdin.
Exit status: 0 = no errors, 1 = errors (or warnings with --strict), 2 = usage.
Requires PyYAML (pip install pyyaml).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass

try:
    import yaml
except ImportError:
    sys.stderr.write("check_manifests.py needs PyYAML: pip install pyyaml\n")
    sys.exit(2)

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}

RULES: dict[str, tuple[str, str]] = {
    "K001": ("error", "Image uses ':latest' or has no tag"),
    "K002": ("info", "Image not pinned by digest"),
    "K003": ("error", "Container has no CPU or memory request"),
    "K004": ("warning", "Container has no memory limit"),
    "K005": ("info", "CPU limit set (can throttle latency-sensitive services)"),
    "K006": ("warning", "Serving container has no readinessProbe"),
    "K007": ("info", "No livenessProbe or startupProbe"),
    "K008": ("warning", "livenessProbe identical to readinessProbe"),
    "K009": ("error", "Pod would violate the 'restricted' Pod Security Standard"),
    "K010": ("warning", "readOnlyRootFilesystem is not true"),
    "K011": ("warning", "Service account token auto-mounted"),
    "K012": ("error", "Workload selector does not match its pod template labels"),
    "K013": ("warning", "Replicated workload has no PodDisruptionBudget"),
    "K014": ("info", "Single replica: any restart or drain is an outage"),
    "K015": ("error", "Removed or deprecated apiVersion"),
    "K016": ("info", "Replicas can land on one node (no spread or anti-affinity)"),
    "K017": ("error", "Secret-looking env var has a literal value"),
    "K018": ("warning", "Service selector matches no workload in the input"),
    "K019": ("info", "Missing app.kubernetes.io/name label"),
    "K020": ("warning", "Uses the 'default' namespace or service account"),
    "K021": ("warning", "Probe port does not match any declared container port"),
    "K022": ("warning", "HPA targets a workload that also sets spec.replicas"),
}

WORKLOAD_KINDS = {"Deployment", "StatefulSet", "DaemonSet", "ReplicaSet", "Job", "CronJob", "Pod"}
REMOVED_APIS = {
    ("extensions/v1beta1", None): "removed; use apps/v1 or networking.k8s.io/v1",
    ("apps/v1beta1", None): "removed in 1.16; use apps/v1",
    ("apps/v1beta2", None): "removed in 1.16; use apps/v1",
    ("networking.k8s.io/v1beta1", None): "removed in 1.22; use networking.k8s.io/v1",
    ("policy/v1beta1", None): "removed in 1.25; use policy/v1 (PodSecurityPolicy has no replacement API: use Pod Security Admission)",
    ("batch/v1beta1", "CronJob"): "removed in 1.25; use batch/v1",
    ("autoscaling/v2beta1", None): "removed in 1.25; use autoscaling/v2",
    ("autoscaling/v2beta2", None): "removed in 1.26; use autoscaling/v2",
    ("rbac.authorization.k8s.io/v1beta1", None): "removed in 1.22; use rbac.authorization.k8s.io/v1",
    ("admissionregistration.k8s.io/v1beta1", None): "removed in 1.22; use v1",
    ("apiextensions.k8s.io/v1beta1", None): "removed in 1.22; use v1",
    ("storage.k8s.io/v1beta1", "CSIStorageCapacity"): "removed in 1.27; use storage.k8s.io/v1",
    ("flowcontrol.apiserver.k8s.io/v1beta2", None): "removed in 1.29; use v1",
    ("flowcontrol.apiserver.k8s.io/v1beta3", None): "removed in 1.32; use v1",
}
SECRET_NAME = re.compile(r"(?i)(passw(or)?d|secret|token|api[_-]?key|private[_-]?key|access[_-]?key|credential|dsn)")
SECRET_OK = re.compile(r"(?i)_(file|path|url_env|dir|name|id)$")
# Matches what the 1.37.1 API server's PodSecurity admission actually allowed when tested.
# "image" volumes were admitted under restricted even though the Pod Security Standards
# page (checked 2026-10-19) does not list them.
PSS_ALLOWED_VOLUMES = {"configMap", "csi", "downwardAPI", "emptyDir", "ephemeral", "image",
                       "persistentVolumeClaim", "projected", "secret"}
BASELINE_CAPS = {"AUDIT_WRITE", "CHOWN", "DAC_OVERRIDE", "FOWNER", "FSETID", "KILL", "MKNOD",
                 "NET_BIND_SERVICE", "SETFCAP", "SETGID", "SETPCAP", "SETUID", "SYS_CHROOT"}


@dataclass
class Finding:
    source: str
    object: str
    rule: str
    severity: str
    message: str
    hint: str = ""


@dataclass
class Obj:
    source: str
    doc: dict

    @property
    def kind(self) -> str:
        return self.doc.get("kind", "")

    @property
    def name(self) -> str:
        return (self.doc.get("metadata") or {}).get("name", "<unnamed>")

    @property
    def namespace(self) -> str | None:
        return (self.doc.get("metadata") or {}).get("namespace")

    @property
    def ref(self) -> str:
        return f"{self.kind}/{self.name}"


# ----------------------------------------------------------------- loading
def load(paths: list[str]) -> list[Obj]:
    objs: list[Obj] = []

    def add_text(text: str, src: str) -> None:
        for doc in yaml.safe_load_all(text):
            if not isinstance(doc, dict):
                continue
            if doc.get("kind") == "List":
                for item in doc.get("items") or []:
                    if isinstance(item, dict):
                        objs.append(Obj(src, item))
            elif doc.get("kind") and doc.get("kind") != "Kustomization":
                objs.append(Obj(src, doc))

    for p in paths:
        if p == "-":
            add_text(sys.stdin.read(), "<stdin>")
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith((".yaml", ".yml")):
                        fp = os.path.join(root, f)
                        with open(fp, encoding="utf-8") as fh:
                            add_text(fh.read(), fp)
        else:
            with open(p, encoding="utf-8") as fh:
                add_text(fh.read(), p)
    return objs


def pod_template(o: Obj) -> tuple[dict, dict] | None:
    """Return (pod metadata, pod spec) for any workload kind."""
    d = o.doc
    if o.kind == "Pod":
        return d.get("metadata") or {}, d.get("spec") or {}
    if o.kind == "CronJob":
        t = (((d.get("spec") or {}).get("jobTemplate") or {}).get("spec") or {}).get("template") or {}
    elif o.kind in WORKLOAD_KINDS:
        t = (d.get("spec") or {}).get("template") or {}
    else:
        return None
    return t.get("metadata") or {}, t.get("spec") or {}


def all_containers(spec: dict) -> list[tuple[str, dict]]:
    out = []
    for key in ("initContainers", "containers", "ephemeralContainers"):
        for c in spec.get(key) or []:
            out.append((key, c))
    return out


def image_state(image: str) -> tuple[bool, bool]:
    has_digest = "@sha256:" in image
    name = image.split("@", 1)[0]
    last = name.rsplit("/", 1)[-1]
    tag = last.split(":", 1)[1] if ":" in last else ""
    return (bool(tag) and tag != "latest") or has_digest, has_digest


def labels_match(selector: dict, labels: dict) -> bool:
    ml = (selector or {}).get("matchLabels") or {}
    if not ml and not (selector or {}).get("matchExpressions"):
        return False
    if any(labels.get(k) != v for k, v in ml.items()):
        return False
    for expr in (selector or {}).get("matchExpressions") or []:
        k, op, vals = expr.get("key"), expr.get("operator"), expr.get("values") or []
        if op == "In" and labels.get(k) not in vals:
            return False
        if op == "NotIn" and labels.get(k) in vals:
            return False
        if op == "Exists" and k not in labels:
            return False
        if op == "DoesNotExist" and k in labels:
            return False
    return True


# ----------------------------------------------------------------- PSS restricted
def pss_restricted_violations(spec: dict) -> list[str]:
    v: list[str] = []
    psc = spec.get("securityContext") or {}
    for f in ("hostNetwork", "hostPID", "hostIPC"):
        if spec.get(f):
            v.append(f"{f}=true")
    for vol in spec.get("volumes") or []:
        kinds = [k for k in vol if k != "name"]
        if kinds and kinds[0] not in PSS_ALLOWED_VOLUMES:
            v.append(f"volume '{vol.get('name')}' uses {kinds[0]}")
    pod_nonroot = psc.get("runAsNonRoot") is True
    pod_seccomp = (psc.get("seccompProfile") or {}).get("type")
    if psc.get("runAsUser") == 0:
        v.append("pod runAsUser=0")
    for _, c in all_containers(spec):
        cn = c.get("name", "?")
        sc = c.get("securityContext") or {}
        if sc.get("privileged"):
            v.append(f"{cn}: privileged=true")
        if sc.get("allowPrivilegeEscalation") is not False:
            v.append(f"{cn}: allowPrivilegeEscalation must be false")
        caps = sc.get("capabilities") or {}
        if "ALL" not in [x.upper() for x in caps.get("drop") or []]:
            v.append(f"{cn}: capabilities.drop must include ALL")
        extra = [x for x in caps.get("add") or [] if x.upper() != "NET_BIND_SERVICE"]
        if extra:
            v.append(f"{cn}: capabilities.add {extra} (only NET_BIND_SERVICE allowed)")
        if sc.get("runAsNonRoot") is False or (not pod_nonroot and sc.get("runAsNonRoot") is not True):
            v.append(f"{cn}: runAsNonRoot must be true (pod or container)")
        if sc.get("runAsUser") == 0:
            v.append(f"{cn}: runAsUser=0")
        st = (sc.get("seccompProfile") or {}).get("type") or pod_seccomp
        if st not in ("RuntimeDefault", "Localhost"):
            v.append(f"{cn}: seccompProfile.type must be RuntimeDefault or Localhost (pod or container)")
        for p in c.get("ports") or []:
            if p.get("hostPort"):
                v.append(f"{cn}: hostPort {p['hostPort']}")
    return v


# ----------------------------------------------------------------- linting
def check(objs: list[Obj]) -> list[Finding]:
    out: list[Finding] = []

    def add(o: Obj, rule: str, msg: str = "", hint: str = "", sev: str | None = None) -> None:
        dsev, dmsg = RULES[rule]
        out.append(Finding(o.source, o.ref, rule, sev or dsev, msg or dmsg, hint))

    pdbs = [o for o in objs if o.kind == "PodDisruptionBudget"]
    hpas = [o for o in objs if o.kind == "HorizontalPodAutoscaler"]
    workloads = [o for o in objs if o.kind in WORKLOAD_KINDS]

    for o in objs:
        api = o.doc.get("apiVersion", "")
        for (gv, kind), why in REMOVED_APIS.items():
            if api == gv and (kind is None or kind == o.kind):
                add(o, "K015", f"{api} {o.kind}: {why}")
        labels = (o.doc.get("metadata") or {}).get("labels") or {}
        if o.kind in WORKLOAD_KINDS | {"Service"} and "app.kubernetes.io/name" not in labels:
            add(o, "K019", hint="Use the recommended app.kubernetes.io/* labels so tools can group objects.")
        if o.namespace == "default":
            add(o, "K020", "Object is placed in the 'default' namespace")

        if o.kind == "Service":
            sel = (o.doc.get("spec") or {}).get("selector") or {}
            if sel and workloads:
                matched = False
                for w in workloads:
                    pt = pod_template(w)
                    if pt and all((pt[0].get("labels") or {}).get(k) == v for k, v in sel.items()):
                        matched = True
                        break
                if not matched:
                    add(o, "K018", f"Service selector {sel} matches no pod template here")

    for w in workloads:
        pt = pod_template(w)
        if pt is None:
            continue
        pmeta, spec = pt
        plabels = pmeta.get("labels") or {}
        wspec = w.doc.get("spec") or {}

        if w.kind in ("Deployment", "StatefulSet", "DaemonSet", "ReplicaSet"):
            if not labels_match(wspec.get("selector") or {}, plabels):
                add(w, "K012", f"selector {wspec.get('selector')} vs template labels {plabels}")

        long_running = w.kind in ("Deployment", "StatefulSet", "DaemonSet", "ReplicaSet")
        if long_running and w.kind != "DaemonSet":
            replicas = wspec.get("replicas", 1)
            hpa = next((h for h in hpas if ((h.doc.get("spec") or {}).get("scaleTargetRef") or {}).get("name") == w.name), None)
            if hpa is not None and "replicas" in wspec:
                add(w, "K022", hint="Drop spec.replicas from the applied manifest once an HPA owns scaling, "
                                    "or every apply resets the replica count.", sev="info")
            effective = max(replicas, ((hpa.doc.get("spec") or {}).get("minReplicas", 1)) if hpa else replicas)
            if effective <= 1:
                add(w, "K014")
            else:
                if not any(labels_match((p.doc.get("spec") or {}).get("selector") or {}, plabels) for p in pdbs):
                    add(w, "K013", hint="Add a policy/v1 PodDisruptionBudget (maxUnavailable: 1) selecting these pods.")
                if not spec.get("topologySpreadConstraints") and not ((spec.get("affinity") or {}).get("podAntiAffinity")):
                    add(w, "K016", hint="Add topologySpreadConstraints on kubernetes.io/hostname (and zone).")

        if spec.get("serviceAccountName") in (None, "default") and spec.get("automountServiceAccountToken") is not False:
            add(w, "K011", "Pod uses the default service account with its token mounted",
                "Set automountServiceAccountToken: false, or create a dedicated ServiceAccount with only the RBAC it needs.")
        elif spec.get("automountServiceAccountToken") is not False:
            add(w, "K011", sev="info",
                hint="Set automountServiceAccountToken: false unless the app calls the Kubernetes API.")
        if spec.get("serviceAccountName") == "default":
            add(w, "K020", "Pod explicitly uses the 'default' service account")

        viol = pss_restricted_violations(spec)
        if viol:
            add(w, "K009", "; ".join(viol),
                "Set pod securityContext runAsNonRoot: true, seccompProfile.type: RuntimeDefault; container "
                "securityContext allowPrivilegeEscalation: false, capabilities.drop: [ALL].")

        for key, c in all_containers(spec):
            if key == "ephemeralContainers":
                continue
            cn = c.get("name", "?")
            image = c.get("image", "")
            ok_tag, digest = image_state(image)
            if not ok_tag:
                add(w, "K001", f"{cn}: image '{image}'")
            elif not digest:
                add(w, "K002", f"{cn}: image '{image}' has a tag but no digest",
                    "Tags are mutable; deploy name:tag@sha256:... from CI for exact rollbacks.")
            res = c.get("resources") or {}
            req, lim = res.get("requests") or {}, res.get("limits") or {}
            restartable_init = key == "initContainers" and c.get("restartPolicy") == "Always"
            if "cpu" not in req or "memory" not in req:
                if not ("cpu" in lim and "memory" in lim):
                    add(w, "K003", f"{cn}: requests={req or '{}'}",
                        "Set cpu and memory requests from observed usage; the scheduler places pods by requests.")
            if "memory" not in lim:
                add(w, "K004", f"{cn}: no limits.memory", "Set limits.memory (often equal to the request).")
            if "cpu" in lim and long_running:
                add(w, "K005", f"{cn}: limits.cpu={lim['cpu']}")
            sc = c.get("securityContext") or {}
            if sc.get("readOnlyRootFilesystem") is not True:
                add(w, "K010", f"{cn}: readOnlyRootFilesystem not set",
                    "Set it true and mount emptyDir at the paths the app writes (/tmp, caches).")
            for env in c.get("env") or []:
                n, v = env.get("name", ""), env.get("value")
                if v not in (None, "") and SECRET_NAME.search(n) and not SECRET_OK.search(n):
                    add(w, "K017", f"{cn}: env {n} has a literal value",
                        "Use valueFrom.secretKeyRef (or a CSI secrets driver).")
            if key == "initContainers" and not restartable_init:
                continue
            if long_running:
                ports = {p.get("containerPort") for p in c.get("ports") or []} | {p.get("name") for p in c.get("ports") or []}
                if c.get("ports") and not c.get("readinessProbe"):
                    add(w, "K006", f"{cn}: no readinessProbe",
                        "Without it, traffic is sent before the app is ready and during shutdown.")
                if not c.get("livenessProbe") and not c.get("startupProbe"):
                    add(w, "K007", f"{cn}: no liveness/startup probe",
                        "Fine if the process exits on fatal errors; add liveness only to catch hangs.")
                lp, rp = c.get("livenessProbe"), c.get("readinessProbe")
                if lp and rp and {k: v for k, v in lp.items() if k in ("httpGet", "tcpSocket", "exec", "grpc")} == \
                        {k: v for k, v in rp.items() if k in ("httpGet", "tcpSocket", "exec", "grpc")}:
                    add(w, "K008", f"{cn}: same check for liveness and readiness",
                        "A dependency outage then restarts every pod. Liveness should check only the process.")
                for pname in ("livenessProbe", "readinessProbe", "startupProbe"):
                    p = c.get(pname) or {}
                    for h in ("httpGet", "tcpSocket", "grpc"):
                        port = (p.get(h) or {}).get("port")
                        if port is not None and c.get("ports") and port not in ports:
                            add(w, "K021", f"{cn}: {pname} port {port!r} not in declared ports")

    out.sort(key=lambda f: (f.source, f.object, -SEVERITY_ORDER[f.severity], f.rule))
    return out


# ----------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    ap.add_argument("--min-severity", choices=SEVERITY_ORDER, default="info")
    ap.add_argument("--ignore", default="", help="comma-separated rule IDs to skip")
    ap.add_argument("--rules", action="store_true")
    a = ap.parse_args(argv)

    if a.rules:
        for rid, (sev, desc) in RULES.items():
            print(f"{rid}  {sev:<7}  {desc}")
        return 0
    if not a.paths:
        ap.print_usage(sys.stderr)
        return 2

    try:
        objs = load(a.paths)
    except (OSError, yaml.YAMLError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    ignore = {r.strip().upper() for r in a.ignore.split(",") if r.strip()}
    floor = SEVERITY_ORDER[a.min_severity]
    findings = [f for f in check(objs) if f.rule not in ignore and SEVERITY_ORDER[f.severity] >= floor]

    if a.json:
        print(json.dumps([asdict(f) for f in findings], indent=2))
    else:
        for f in findings:
            print(f"{f.source}  {f.object}  {f.severity.upper():<7} {f.rule}  {f.message}")
            if f.hint:
                print(f"    -> {f.hint}")
        counts = {s: sum(1 for f in findings if f.severity == s) for s in SEVERITY_ORDER}
        print(f"\n{len(objs)} object(s); {counts['error']} error(s), {counts['warning']} warning(s), {counts['info']} info")

    worst = max((SEVERITY_ORDER[f.severity] for f in findings), default=-1)
    return 1 if worst >= 2 or (a.strict and worst >= 1) else 0


if __name__ == "__main__":
    sys.exit(main())
