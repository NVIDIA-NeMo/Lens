#!/usr/bin/env python3
"""Convert the per-node OTel-collector zstd archives (OTLP JSON) into the per-rank console-format
lens_rank{N}.jsonl.gz files that json_to_perfetto.py / goodput_report.py already ingest.

The collector archives are per-NODE and OTLP-encoded (resourceSpans/scopeSpans, unix-nano times,
attribute lists); the analysis tools expect per-RANK console ReadableSpan.to_json (name, ISO
start/end, attributes dict). This is the downstream-ingest adapter. All node files for a job
(rotated backups + current, across every requeue attempt) are merged, then split by rank -> one
lens_rank{N}.jsonl.gz per rank with EVERY attempt's spans (matching the requeue lens-dir model).

Usage: otel_archive_to_lens.py <archive_dir> <out_lens_dir>
"""
import glob
import gzip
import json
import os
import subprocess
import sys
from datetime import datetime


def _av(v):
    """OTLP AnyValue -> python scalar."""
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return v["doubleValue"]
    if "boolValue" in v:
        return v["boolValue"]
    return json.dumps(v)


def _attrs(lst):
    return {a["key"]: _av(a["value"]) for a in (lst or [])}


def _iso(nanos):
    """unix nanoseconds (str) -> naive-local ISO (lens_analysis.ts parses it back to the same epoch)."""
    return datetime.fromtimestamp(int(nanos) / 1e9).isoformat()


def convert(archive_dir, out_dir):
    files = sorted(glob.glob(os.path.join(archive_dir, "*.zst")))
    if not files:
        print(f"no .zst archives under {archive_dir}", file=sys.stderr)
        return 1
    os.makedirs(out_dir, exist_ok=True)
    # (role, rank) -> open gzip writer
    writers, counts, instance_ids = {}, {}, set()

    def writer_for(role, rank):
        key = (role, rank)
        if key not in writers:
            fn = (f"lens_rank{rank}_ckptworker.jsonl.gz" if role == "ckptworker"
                  else f"lens_rank{rank}.jsonl.gz")
            writers[key] = gzip.open(os.path.join(out_dir, fn), "wt")
            counts[key] = 0
        return writers[key]

    for f in files:
        raw = subprocess.run(["zstd", "-dc", f], capture_output=True).stdout.decode("utf-8", "replace")
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn tail from an ungraceful kill
            for rspan in obj.get("resourceSpans", []):
                res = _attrs(rspan.get("resource", {}).get("attributes"))
                iid = str(res.get("service.instance.id", ""))
                instance_ids.add(iid)
                rank = res.get("dl.rank")
                if rank is None and "-rank" in iid:
                    rank = iid.rsplit("-rank", 1)[1]
                rank = int(rank) if rank is not None and str(rank).isdigit() else 0
                role = "ckptworker" if "ckptworker" in iid.lower() else "trainer"
                w = writer_for(role, rank)
                for sspan in rspan.get("scopeSpans", []):
                    for sp in sspan.get("spans", []):
                        rec = {
                            "name": sp["name"],
                            "context": {"trace_id": sp.get("traceId", ""), "span_id": sp.get("spanId", "")},
                            "parent_id": sp.get("parentSpanId") or None,
                            "start_time": _iso(sp["startTimeUnixNano"]),
                            "end_time": _iso(sp["endTimeUnixNano"]),
                            "attributes": _attrs(sp.get("attributes")),
                            "resource": {"attributes": res},
                        }
                        w.write(json.dumps(rec) + "\n")
                        counts[(role, rank)] += 1

    for w in writers.values():
        w.close()
    print(f"[otel->lens] {len(files)} archive files -> {out_dir}")
    print(f"[otel->lens] instance ids: {sorted(x for x in instance_ids if x)}")
    for (role, rank), n in sorted(counts.items()):
        print(f"  lens_rank{rank}{'_ckptworker' if role=='ckptworker' else ''}: {n} spans")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: otel_archive_to_lens.py <archive_dir> <out_lens_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(convert(sys.argv[1], sys.argv[2]))
