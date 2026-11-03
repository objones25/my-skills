"""Parse the dense-model table in references/reference-configs.md."""
from __future__ import annotations

from pathlib import Path


def dense_rows(skill_root: Path) -> list[dict]:
    text = (skill_root / "references" / "reference-configs.md").read_text()
    table = text.split("## Dense models")[1].split("Every model from SmolLM2")[0]
    rows = []
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 14 or cells[0] in ("Model",) or cells[0].startswith("---"):
            continue
        params = float(cells[1][:-1]) * (1e9 if cells[1].endswith("B") else 1e6)
        d, layers, vocab = int(cells[2]), int(cells[3]), int(cells[8])
        rows.append({"name": cells[0], "params": params, "aspect": d / layers,
                     "share": vocab * d / params, "tied": cells[11] == "yes"})
    return rows
