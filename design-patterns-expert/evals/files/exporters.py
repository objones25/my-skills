"""Weekly report exporter. Grew one format at a time; a fourth (xlsx) is planned."""
import csv
import io
import json
import smtplib
from dataclasses import dataclass


@dataclass
class ReportRow:
    region: str
    revenue_cents: int


def export_report(rows: list[ReportRow], kind: str) -> str:
    if kind == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["region", "revenue_cents"])
        for r in rows:
            writer.writerow([r.region, r.revenue_cents])
        body = buf.getvalue()
    elif kind == "json":
        body = json.dumps([{"region": r.region, "revenue_cents": r.revenue_cents} for r in rows])
    elif kind == "markdown":
        lines = ["| region | revenue_cents |", "|---|---|"]
        lines += [f"| {r.region} | {r.revenue_cents} |" for r in rows]
        body = "\n".join(lines)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    server = smtplib.SMTP("smtp.internal", 25)
    server.sendmail("reports@example.com", ["finance@example.com"], body)
    server.quit()
    return body
