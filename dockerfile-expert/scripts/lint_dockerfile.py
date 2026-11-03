#!/usr/bin/env python3
"""Dependency-free Dockerfile linter focused on production best practices.

It complements, not replaces, `docker build --check` (BuildKit build checks)
and hadolint. Those tools catch syntax-level issues; this script targets the
structural problems they miss: root in the final stage, single-stage builds
that ship toolchains, whole-context COPY before dependency install, secrets in
ENV/ARG/COPY, unpinned images in COPY --from, and so on.

Usage:
    lint_dockerfile.py [--json] [--strict] [--min-severity LEVEL]
                       [--context DIR] [--ignore DF001,DF003] Dockerfile...
    lint_dockerfile.py --rules

Exit status: 0 = no errors, 1 = errors (or warnings with --strict), 2 = usage.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}

RULES: dict[str, tuple[str, str]] = {
    "DF001": ("warning", "Missing '# syntax=docker/dockerfile:1' parser directive"),
    "DF002": ("warning", "Base image uses ':latest' or has no tag"),
    "DF003": ("info", "Base image not pinned by digest"),
    "DF004": ("error", "Final stage runs as root"),
    "DF005": ("warning", "Final USER is a name, not a numeric UID"),
    "DF006": ("error", "'apt-get update' without 'apt-get install' in the same RUN"),
    "DF007": ("warning", "'apt-get install' without '--no-install-recommends'"),
    "DF008": ("warning", "'apt-get install' without '-y'"),
    "DF009": ("info", "apt lists neither removed nor on a cache mount"),
    "DF010": ("warning", "ADD used for local files; use COPY"),
    "DF011": ("error", "Secret-looking name in ENV/ARG"),
    "DF012": ("warning", "Remote script piped into a shell"),
    "DF013": ("warning", "Shell-form CMD/ENTRYPOINT"),
    "DF014": ("warning", "Whole build context copied before dependency install"),
    "DF015": ("warning", "pip install without --no-cache-dir or a cache mount"),
    "DF016": ("warning", "'npm install' in a build; use 'npm ci'"),
    "DF017": ("info", "No HEALTHCHECK (only Docker/Compose use it; Kubernetes ignores it)"),
    "DF018": ("warning", "Relative WORKDIR"),
    "DF019": ("warning", "sudo used in a Dockerfile"),
    "DF020": ("warning", "World-writable permissions (777 / a+rwx)"),
    "DF021": ("warning", "No .dockerignore in the build context"),
    "DF022": ("info", "Pipe in RUN without 'set -o pipefail'"),
    "DF023": ("warning", "MAINTAINER is deprecated; use a LABEL"),
    "DF024": ("warning", "Multiple CMD or ENTRYPOINT in one stage (only the last applies)"),
    "DF025": ("warning", "'apk add' without '--no-cache' or a cache mount"),
    "DF026": ("warning", "Final stage is based on a build toolchain or full-size image"),
    "DF027": ("warning", "Final stage installs compilers or -dev packages"),
    "DF028": ("warning", "COPY --from an external image without a pinned tag"),
    "DF029": ("info", "DEBIAN_FRONTEND set with ENV persists into the runtime image"),
    "DF030": ("warning", "Single-stage build that also compiles or bundles"),
    "DF031": ("error", "COPY/ADD of a likely secret file"),
    "DF032": ("warning", "Process started through npm/yarn/pnpm (signals not forwarded)"),
}

SECRET_NAME = re.compile(
    r"(?i)(passw(or)?d|secret|token|api[_-]?key|private[_-]?key|access[_-]?key|credential)"
)
SECRET_NAME_OK_SUFFIX = re.compile(r"(?i)_(file|path|url|dir|name|id)$")
SECRET_FILES = re.compile(
    r"(?i)(^|/)(\.env(\.[\w-]+)?|id_rsa|id_ed25519|\.npmrc|\.netrc|\.pypirc|"
    r"credentials(\.json)?|[\w.-]+\.(pem|key|p12|pfx))$"
)
DEP_INSTALL = re.compile(
    r"\b(pip3?\s+install|python3?\s+-m\s+pip\s+install|uv\s+(sync|pip\s+install)|"
    r"poetry\s+install|pipenv\s+install|npm\s+(ci|install|i)\b|yarn(\s+install)?\s*($|&&|;)|"
    r"pnpm\s+(install|i)\b|go\s+mod\s+download|bundle\s+install|cargo\s+fetch|"
    r"composer\s+install|mvn\b.*dependency:go-offline)"
)
BUILD_STEP = re.compile(
    r"\b(go\s+build|cargo\s+build|npm\s+run\s+build|pnpm\s+(run\s+)?build|yarn\s+(run\s+)?build|"
    r"tsc\b|mvn\s+(package|install)|gradle\w*\s+(build|assemble|jar)|dotnet\s+publish|"
    r"pip3?\s+wheel|make\b|gcc\b|cmake\b)"
)
TOOLCHAIN_BASE = re.compile(
    r"(^|/)(golang|rust|maven|gradle|buildpack-deps)(:|@|$)|/dotnet/sdk(:|@|$)"
)
FULL_SIZE_BASE = re.compile(r"(^|/)(python|node|ruby|php)(:[\w.-]+)?(@|$)")
TOOLCHAIN_PKGS = re.compile(r"(?<![\w.-])(build-essential|gcc|g\+\+|clang|make|cmake|[\w.+-]+-dev)(?![\w.-])")
NONROOT_BASE_HINT = re.compile(r"nonroot|distroless/.*:nonroot|^dhi\.io/(?!.*-dev)")


@dataclass
class Finding:
    file: str
    line: int
    rule: str
    severity: str
    message: str
    hint: str = ""


@dataclass
class Instr:
    line: int
    cmd: str
    args: str
    heredoc: str = ""

    @property
    def text(self) -> str:
        return f"{self.args}\n{self.heredoc}" if self.heredoc else self.args


@dataclass
class Stage:
    index: int
    base: str
    name: str | None
    line: int
    instrs: list[Instr] = field(default_factory=list)


# ----------------------------------------------------------------- parsing
def parse(text: str) -> tuple[dict[str, str], list[Instr]]:
    lines = text.splitlines()
    directives: dict[str, str] = {}
    i = 0
    while i < len(lines):
        m = re.match(r"^\s*#\s*([a-zA-Z]+)\s*=\s*(.+?)\s*$", lines[i])
        if not m:
            break
        directives[m.group(1).lower()] = m.group(2)
        i += 1
    escape = directives.get("escape", "\\")

    instrs: list[Instr] = []
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        start = i + 1
        parts = []
        cur = raw
        while True:
            body = cur.rstrip()
            if body.endswith(escape):
                parts.append(body[: -len(escape)])
                i += 1
                while i < len(lines) and lines[i].strip().startswith("#"):
                    i += 1
                if i >= len(lines):
                    break
                cur = lines[i]
            else:
                parts.append(body)
                break
        logical = " ".join(p.strip() for p in parts)
        i += 1
        m = re.match(r"^(\w+)\s*(.*)$", logical)
        if not m:
            continue
        cmd, args = m.group(1).upper(), m.group(2)
        heredoc = ""
        for hm in re.finditer(r"<<(-?)([\"']?)(\w+)\2", args):
            dash, delim = hm.group(1), hm.group(3)
            body_lines = []
            while i < len(lines):
                ln = lines[i]
                i += 1
                if (ln.strip() if dash else ln.rstrip()) == delim:
                    break
                body_lines.append(ln)
            heredoc += "\n".join(body_lines) + "\n"
        instrs.append(Instr(start, cmd, args, heredoc))
    return directives, instrs


def split_flags(args: str) -> tuple[dict[str, list[str]], str]:
    flags: dict[str, list[str]] = {}
    rest = args
    while True:
        m = re.match(r"^--([\w-]+)(?:=(\S+))?\s*", rest)
        if not m:
            break
        flags.setdefault(m.group(1), []).append(m.group(2) or "")
        rest = rest[m.end():]
    return flags, rest


def substitute(value: str, env: dict[str, str]) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1) or m.group(3)
        default = m.group(2)
        if name in env and env[name] != "":
            return env[name]
        if default is not None:
            return default
        return m.group(0)

    return re.sub(r"\$\{(\w+)(?::-([^}]*))?\}|\$(\w+)", repl, value)


def build_stages(instrs: list[Instr]) -> tuple[list[Stage], dict[str, str]]:
    global_args: dict[str, str] = {}
    stages: list[Stage] = []
    for ins in instrs:
        if ins.cmd == "ARG" and not stages:
            for tok in ins.args.split():
                k, _, v = tok.partition("=")
                global_args[k] = v.strip("\"'")
        if ins.cmd == "FROM":
            _, rest = split_flags(ins.args)
            toks = rest.split()
            base = substitute(toks[0], global_args) if toks else ""
            name = toks[2].lower() if len(toks) >= 3 and toks[1].lower() == "as" else None
            stages.append(Stage(len(stages), base, name, ins.line))
        elif stages:
            stages[-1].instrs.append(ins)
    return stages, global_args


# ----------------------------------------------------------------- helpers
def is_json_form(args: str) -> bool:
    a = args.strip()
    if not (a.startswith("[") and a.endswith("]")):
        return False
    try:
        return isinstance(json.loads(a), list)
    except json.JSONDecodeError:
        return False


def image_tag_state(ref: str) -> tuple[bool, bool]:
    """Return (has_usable_tag, has_digest)."""
    has_digest = "@sha256:" in ref
    name = ref.split("@", 1)[0]
    last = name.rsplit("/", 1)[-1]
    tag = last.split(":", 1)[1] if ":" in last else ""
    usable = bool(tag) and tag != "latest"
    return (usable or has_digest, has_digest)


def run_body(ins: Instr) -> str:
    _, rest = split_flags(ins.args)
    return f"{rest}\n{ins.heredoc}" if ins.heredoc else rest


def cache_mounts(ins: Instr) -> list[str]:
    flags, _ = split_flags(ins.args)
    return [m for m in flags.get("mount", []) if "type=cache" in m]


def resolve_user(stages: list[Stage], idx: int, seen: set[int] | None = None) -> tuple[str | None, str]:
    """Effective USER of stage idx and the base it ultimately rests on."""
    seen = seen or set()
    stage = stages[idx]
    for ins in reversed(stage.instrs):
        if ins.cmd == "USER":
            return ins.args.strip(), stage.base
    names = {s.name: s.index for s in stages if s.name}
    parent = names.get(stage.base.lower())
    if parent is not None and parent not in seen and parent < idx:
        seen.add(idx)
        return resolve_user(stages, parent, seen)
    return None, stage.base


def external_base(stage: Stage, stages: list[Stage]) -> bool:
    names = {s.name for s in stages[: stage.index] if s.name}
    return stage.base.lower() not in names and stage.base.lower() != "scratch"


# ----------------------------------------------------------------- linting
def lint(path: str, text: str, context_dir: str | None) -> list[Finding]:
    out: list[Finding] = []

    def add(line: int, rule: str, msg: str = "", hint: str = "") -> None:
        sev, default_msg = RULES[rule]
        out.append(Finding(path, line, rule, sev, msg or default_msg, hint))

    directives, instrs = parse(text)
    stages, global_args = build_stages(instrs)
    if not stages:
        add(1, "DF001", "No FROM instruction found")
        return out

    if "syntax" not in directives:
        add(1, "DF001", hint="Add '# syntax=docker/dockerfile:1' as the first line to get the "
                             "current stable Dockerfile frontend (heredocs, mounts, checks).")

    final = stages[-1]
    stage_names = {s.name for s in stages if s.name}
    shell_has_pipefail = False

    for st in stages:
        if external_base(st, stages):
            if "$" in st.base:
                pass
            else:
                usable, digest = image_tag_state(st.base)
                if not usable:
                    add(st.line, "DF002", f"Base image '{st.base}' is unpinned (':latest' or no tag)",
                        "Pin at least a minor version tag, e.g. python:3.14-slim-trixie.")
                elif not digest:
                    add(st.line, "DF003", f"'{st.base}' is not pinned by digest",
                        "For reproducible builds use name:tag@sha256:<digest> and let "
                        "Dependabot/Renovate bump it.")

        cmd_lines, ep_lines = [], []
        copied_whole_context_at: int | None = None
        installed_deps_before_copy = False
        for ins in st.instrs:
            c = ins.cmd
            if c == "SHELL" and "pipefail" in ins.args:
                shell_has_pipefail = True
            if c == "MAINTAINER":
                add(ins.line, "DF023", hint="Use LABEL org.opencontainers.image.authors=...")
            if c == "WORKDIR":
                wd = ins.args.strip().strip("\"'")
                if not (wd.startswith("/") or wd.startswith("$")):
                    add(ins.line, "DF018", f"WORKDIR '{wd}' is relative")
            if c in ("CMD", "ENTRYPOINT"):
                (cmd_lines if c == "CMD" else ep_lines).append(ins.line)
                if not is_json_form(ins.args):
                    add(ins.line, "DF013", f"{c} uses shell form",
                        f'Use exec form, e.g. {c} ["executable", "arg"]. Shell form makes '
                        "/bin/sh PID 1, which does not forward SIGTERM.")
                if re.search(r'^\[?\s*"?(npm|yarn|pnpm)"?\s*,?\s*"?(run\b|start\b)', ins.args.strip().lstrip("[")):
                    add(ins.line, "DF032", hint='Run the runtime directly, e.g. CMD ["node", "dist/server.js"].')
            if c in ("ENV", "ARG"):
                pairs = re.findall(r"([\w.-]+)(?:=(\"[^\"]*\"|'[^']*'|\S*))?", ins.args)
                for k, v in pairs:
                    if SECRET_NAME.search(k) and not SECRET_NAME_OK_SUFFIX.search(k):
                        if c == "ENV" and v:
                            add(ins.line, "DF011", f"ENV '{k}' bakes a secret-looking value into the image",
                                "Use RUN --mount=type=secret,id=...,env=NAME at build time and inject "
                                "runtime secrets from the orchestrator.")
                        elif c == "ARG":
                            out.append(Finding(path, ins.line, "DF011", "warning",
                                               f"ARG '{k}' looks like a secret; build args are recorded in image history",
                                               "Use RUN --mount=type=secret instead."))
                    if k == "DEBIAN_FRONTEND" and c == "ENV":
                        add(ins.line, "DF029", hint="Use ARG DEBIAN_FRONTEND=noninteractive or set it inline in the RUN.")
            if c in ("COPY", "ADD"):
                flags, rest = split_flags(ins.args)
                srcs = rest.split()[:-1] if not is_json_form(rest) else json.loads(rest)[:-1]
                for s in srcs:
                    if SECRET_FILES.search(s.rstrip("/")):
                        add(ins.line, "DF031", f"{c} of '{s}' looks like a secret file",
                            "Use a secret mount; also exclude it in .dockerignore.")
                if c == "COPY" and "from" in flags:
                    ref = flags["from"][0]
                    if ref.lower() not in stage_names and not ref.isdigit() and ("/" in ref or ":" in ref):
                        usable, _ = image_tag_state(ref)
                        if not usable:
                            add(ins.line, "DF028", f"COPY --from={ref} floats with upstream releases",
                                "Pin an exact version, e.g. ghcr.io/astral-sh/uv:0.13.3.")
                if c == "ADD":
                    for s in srcs:
                        if not re.match(r"^(https?://|git@|[\w.-]+://)", s):
                            if re.search(r"\.(tar(\.\w+)?|tgz|txz)$", s):
                                continue
                            add(ins.line, "DF010", f"ADD '{s}' copies a local path",
                                "COPY is explicit; ADD also auto-extracts archives.")
                if c == "COPY" and "from" not in flags and any(s in (".", "./") for s in srcs):
                    if copied_whole_context_at is None and not installed_deps_before_copy:
                        copied_whole_context_at = ins.line
            if c == "RUN":
                body = run_body(ins)
                if is_json_form(split_flags(ins.args)[1]):
                    body = " ".join(json.loads(split_flags(ins.args)[1]))
                mounts = cache_mounts(ins)
                if DEP_INSTALL.search(body):
                    if copied_whole_context_at is not None and not installed_deps_before_copy:
                        add(copied_whole_context_at, "DF014",
                            "COPY of the whole context precedes the dependency install, so any source "
                            "edit reinstalls every dependency",
                            "Copy or bind-mount only the lockfile(s) first, install, then COPY the source.")
                        copied_whole_context_at = None
                    if copied_whole_context_at is None:
                        installed_deps_before_copy = True
                if re.search(r"\bapt(-get)?\s+update\b", body) and not re.search(r"\bapt(-get)?\s+(-\S+\s+)*install\b", body):
                    add(ins.line, "DF006", hint="Combine: RUN apt-get update && apt-get install -y --no-install-recommends ...")
                if re.search(r"\bapt(-get)?\s+(-\S+\s+)*install\b", body):
                    if "--no-install-recommends" not in body and "Install-Recommends=false" not in body:
                        add(ins.line, "DF007")
                    if not re.search(r"(\s-\w*y\w*\b|--yes|--assume-yes|-qq)", body):
                        add(ins.line, "DF008")
                    if "/var/lib/apt/lists" not in body and not any("/var/lib/apt" in m or "/var/cache/apt" in m for m in mounts):
                        add(ins.line, "DF009", hint="End with '&& rm -rf /var/lib/apt/lists/*' or use cache mounts "
                                                    "for /var/cache/apt and /var/lib/apt.")
                    if st is final and TOOLCHAIN_PKGS.search(body):
                        add(ins.line, "DF027", hint="Install toolchains in a builder stage and copy only artifacts.")
                if re.search(r"\bapk\s+add\b", body) and "--no-cache" not in body and not any("/var/cache/apk" in m or "/etc/apk/cache" in m for m in mounts):
                    add(ins.line, "DF025")
                if re.search(r"\b(pip3?|python3?\s+-m\s+pip)\s+install\b", body) and not re.search(r"\buv\s+pip", body):
                    if "--no-cache-dir" not in body and not any("pip" in m for m in mounts):
                        if not any(i2.cmd == "ENV" and "PIP_NO_CACHE_DIR" in i2.args for s2 in stages for i2 in s2.instrs):
                            add(ins.line, "DF015")
                if re.search(r"\bnpm\s+(install|i)\b(?!\s+(-g|--global))", body):
                    add(ins.line, "DF016", hint="npm ci installs exactly what package-lock.json says and fails if it is stale.")
                if re.search(r"\b(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(ba|z|da)?sh\b", body):
                    add(ins.line, "DF012", hint="Download to a file, verify a checksum (ADD --checksum=sha256:...), then run it.")
                if re.search(r"(^|[\s;&|])sudo\s", body):
                    add(ins.line, "DF019", hint="RUN steps already run as the current USER; switch with USER instead.")
                if re.search(r"chmod\s+(-R\s+)?(0?777|a\+rwx|ugo\+rwx)", body):
                    add(ins.line, "DF020")
                if re.search(r"(?<!\|)\|(?!\|)", re.sub(r"'[^']*'|\"[^\"]*\"", "", body)):
                    if "pipefail" not in body and not shell_has_pipefail:
                        add(ins.line, "DF022", hint="Prefix with 'set -o pipefail &&' (needs bash; Debian sh is dash) "
                                                    "or set SHELL [\"/bin/bash\", \"-o\", \"pipefail\", \"-c\"].")
        for lines_ in (cmd_lines, ep_lines):
            if len(lines_) > 1:
                add(lines_[-1], "DF024")

    # ---- final stage checks
    user, root_base = resolve_user(stages, final.index)
    if user is None:
        if not NONROOT_BASE_HINT.search(root_base) and root_base.lower() != "scratch":
            add(final.line, "DF004", "Final stage never sets USER, so it inherits root from the base image",
                "Create a user with a fixed numeric UID and add 'USER 10001:10001'.")
        elif root_base.lower() == "scratch":
            add(final.line, "DF004", "Final stage is FROM scratch with no USER (runs as UID 0)",
                "Add 'USER 65532:65532' (no /etc/passwd entry is needed for a numeric UID).")
    else:
        name = user.split(":", 1)[0]
        if name in ("root", "0"):
            add(final.line, "DF004", f"Final stage USER is '{user}'")
        elif not name.isdigit() and not name.startswith("$"):
            add(final.line, "DF005", f"USER '{user}' is not numeric",
                "Kubernetes runAsNonRoot can only verify numeric UIDs; use e.g. 'USER 10001:10001'.")

    if external_base(final, stages) and "$" not in final.base:
        if TOOLCHAIN_BASE.search(final.base):
            add(final.line, "DF026", f"Final image '{final.base}' is a full build toolchain",
                "Build in an earlier stage; ship a slim, distroless, or static runtime image.")
        elif FULL_SIZE_BASE.search(final.base) and not re.search(r"slim|alpine|distroless", final.base):
            out.append(Finding(path, final.line, "DF026", "info",
                               f"Final image '{final.base}' is the full-size variant",
                               "Prefer the -slim variant unless you need its extra system libraries."))

    if len(stages) == 1:
        for ins in final.instrs:
            if ins.cmd == "RUN" and BUILD_STEP.search(run_body(ins)):
                add(ins.line, "DF030", hint="Move compilation into a builder stage and COPY --from=builder the outputs.")
                break

    if not any(i.cmd == "HEALTHCHECK" for i in final.instrs):
        add(final.line, "DF017")

    if context_dir is not None and not os.path.exists(os.path.join(context_dir, ".dockerignore")):
        base = os.path.basename(path)
        if not os.path.exists(os.path.join(context_dir, base + ".dockerignore")):
            add(1, "DF021", f"No .dockerignore in {context_dir}",
                "Exclude .git, .env*, virtualenvs, node_modules, build outputs, and data.")

    out.sort(key=lambda f: (f.line, -SEVERITY_ORDER[f.severity], f.rule))
    return out


# ----------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    ap.add_argument("--min-severity", choices=SEVERITY_ORDER, default="info")
    ap.add_argument("--context", help="build context dir (default: the Dockerfile's dir)")
    ap.add_argument("--ignore", default="", help="comma-separated rule IDs to skip")
    ap.add_argument("--rules", action="store_true", help="list rules and exit")
    a = ap.parse_args(argv)

    if a.rules:
        for rid, (sev, desc) in RULES.items():
            print(f"{rid}  {sev:<7}  {desc}")
        return 0
    if not a.files:
        ap.print_usage(sys.stderr)
        return 2

    ignore = {r.strip().upper() for r in a.ignore.split(",") if r.strip()}
    floor = SEVERITY_ORDER[a.min_severity]
    findings: list[Finding] = []
    for f in a.files:
        try:
            with open(f, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            print(f"{f}: {e}", file=sys.stderr)
            return 2
        ctx = a.context if a.context is not None else (os.path.dirname(os.path.abspath(f)))
        findings += lint(f, text, ctx)

    findings = [x for x in findings if x.rule not in ignore and SEVERITY_ORDER[x.severity] >= floor]
    if a.json:
        print(json.dumps([asdict(x) for x in findings], indent=2))
    else:
        for x in findings:
            print(f"{x.file}:{x.line}  {x.severity.upper():<7} {x.rule}  {x.message}")
            if x.hint:
                print(f"    -> {x.hint}")
        counts = {s: sum(1 for x in findings if x.severity == s) for s in SEVERITY_ORDER}
        print(f"\n{counts['error']} error(s), {counts['warning']} warning(s), {counts['info']} info")

    worst = max((SEVERITY_ORDER[x.severity] for x in findings), default=-1)
    if worst >= SEVERITY_ORDER["error"] or (a.strict and worst >= SEVERITY_ORDER["warning"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
