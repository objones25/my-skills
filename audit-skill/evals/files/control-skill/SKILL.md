---
name: control-skill
description: "Tiny fixture skill for audit-skill's control eval. Use when testing audit-skill. Not for real use."
---

# Control skill

- `json.dumps(obj, sort_keys=True)` writes object keys in sorted order.
- `pathlib.PurePath.is_relative_to` was added in Python 3.9.
- `subprocess.run` raises `subprocess.TimeoutExpired` when its `timeout` expires.
