"""Shared span-ingest plumbing for the lens analysis tools (goodput_report.py,
json_to_perfetto.py, and any future dashboard).

The point is that every tool reads the SAME run files the SAME way -- one tolerant
reader, one timestamp convention, one file/role/rank discovery -- so a change (a new
file name, the reckoner's lens_slurm output, the workload/pre_startup rename) is made
once here instead of drifting across scripts. Interpretation that is specific to a tool
(goodput's CLASSIFY taxonomy, lost-work rule, etc.) stays in that tool; this is just the
ingest layer they all share.
"""
import glob
import gzip
import json
import os
import re
from datetime import datetime


def read_objects(path):
    """Read a lens JSONL dump (plain or gzip) -> list of JSON objects, tolerating an
    incomplete/trailerless gzip stream: a scancel'd/crashed job flushes but never closes
    the file, so gunzip hits EOFError at the missing trailer -- everything before the last
    flush is still valid, so we keep the objects that decode and stop at the first bad one."""
    data = ""
    try:
        opener = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
        with opener as f:
            for line in f:
                data += line
    except (EOFError, OSError):
        pass  # incomplete final gzip block -- keep the flushed lines we got
    dec = json.JSONDecoder()
    i, n, out = 0, len(data), []
    while i < n:
        while i < n and data[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            obj, i = dec.raw_decode(data, i)
        except json.JSONDecodeError:
            break  # truncated final line from an incomplete flush
        out.append(obj)
    return out


def read_spans(path):
    """read_objects filtered to span records (a dict with a 'name'). Drops metric records."""
    return [o for o in read_objects(path) if isinstance(o, dict) and o.get("name")]


def ts(iso):
    """lens ISO timestamp -> unix seconds. sacct/OTel naive-local & UTC-'Z' both land on
    the same epoch line, so backdated reckoner spans align with the training's time.time()."""
    if iso is None:
        return None
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def us(iso):
    """... -> unix MICROseconds (perfetto's unit)."""
    t = ts(iso)
    return None if t is None else t * 1_000_000.0


def role_of(basename):
    """Classify a lens span file by name: 'slurm' (reckoner job-envelope spans), 'ckptworker'
    (async checkpoint worker), 'trainer' (a rank), or 'other'."""
    if basename.startswith("lens_slurm"):
        return "slurm"
    if "_ckptworker" in basename:
        return "ckptworker"
    if basename.startswith("lens_rank"):
        return "trainer"
    return "other"


def rank_of(basename, fallback=None):
    """Pull the rank number out of a file name (lens_rank3.jsonl[.gz], lens_rank3_ckptworker...)."""
    m = re.search(r"rank(\d+)", basename)
    return int(m.group(1)) if m else fallback


# ---- shared INTERPRETATION: which iterations were lost work --------------------------
# One source of truth for "is this iteration redone/lost", so the goodput numbers and the
# perfetto coloring can't disagree. Spans-only, per-run rule (see goodput_report for prose).
ITER_SPAN = "megatron.train.iteration"
_CKPT_SPANS = ("megatron.checkpoint.exposed_save", "megatron.checkpoint.save")


def iteration_number(o):
    v = (o.get("attributes") or {}).get("megatron.iteration")
    return int(v) if v is not None else None


def iteration_status(spans):
    """For ONE rank's spans, classify every megatron.train.iteration span. Returns
    {id(span): 'committed' | 'redone' | 'uncommitted'}. Rule: a requeue reloads an earlier
    checkpoint, so the iteration number jumps BACKWARD -> a new run; within each run, an
    iteration AFTER that run's last durable checkpoint is lost (redone if a later run exists,
    uncommitted for the final run's unsaved tail); a run that never checkpoints loses all."""
    iters = [(o, iteration_number(o)) for o in spans if o.get("name") == ITER_SPAN]
    iters = [(o, it) for o, it in iters if it is not None]
    if not iters:
        return {}
    iters_by_t = sorted(iters, key=lambda x: ts(x[0]["start_time"]))
    runs, prev_it = [], None
    for (o, it) in iters_by_t:
        if not runs or (prev_it is not None and it <= prev_it):     # backward jump = new run
            runs.append({"iters": [], "t0": ts(o["start_time"]), "t1": ts(o["end_time"])})
        runs[-1]["iters"].append((o, it))
        runs[-1]["t1"] = max(runs[-1]["t1"], ts(o["end_time"]))
        prev_it = it
    ckpt_at = [(ts(o["start_time"]), iteration_number(o)) for o in spans
               if o.get("name") in _CKPT_SPANS and iteration_number(o) is not None]
    out = {}
    for ri, run in enumerate(runs):
        inrun = [ci for (ct, ci) in ckpt_at if run["t0"] <= ct <= run["t1"] + 1.0]
        lc = max(inrun) if inrun else None
        is_final = (ri == len(runs) - 1)
        for (o, it) in run["iters"]:
            out[id(o)] = ("committed" if (lc is not None and it <= lc)
                          else "uncommitted" if is_final else "redone")
    return out


def discover(dirs):
    """Yield (path, role, rank) for every lens *.jsonl[.gz] under each input dir (files may
    also be passed directly). One place that knows the run-dir layout, shared by all tools."""
    for inp in dirs:
        if os.path.isdir(inp):
            files = sorted(glob.glob(os.path.join(inp, "*.jsonl")) +
                           glob.glob(os.path.join(inp, "*.jsonl.gz")))
        else:
            files = sorted(glob.glob(inp))
        for f in files:
            base = os.path.basename(f)
            yield f, role_of(base), rank_of(base)
