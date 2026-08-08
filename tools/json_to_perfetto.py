#!/usr/bin/env python3
"""Convert lens_json OTel span/metric dumps into a Perfetto/Chrome trace JSON file.

Usage:
    python3 json_to_perfetto.py <input_dir_or_files...> -o trace.json

Input files are expected to contain back-to-back pretty-printed JSON objects
(not strict JSONL) that are either OTel spans (have "context"/"start_time")
or OTel resource_metrics dumps (have "resource_metrics").

Output is the legacy Chrome Trace Event Format, which Perfetto
(https://ui.perfetto.dev) and chrome://tracing both load directly.
"""
import argparse
import json
import os
import sys
# Shared ingest plumbing (one reader / timestamp / file-discovery for all lens tools).
from lens_analysis import read_objects, read_spans, ts, us as iso_to_us, discover, iteration_status


def rank_of(resource_attrs, fallback):
    r = resource_attrs.get("dl.rank")
    return int(r) if r is not None else fallback


def flatten_attrs(attrs):
    # Chrome trace "args" must be JSON-serializable scalars/containers; OTel
    # attribute values already are, so just pass them through.
    return {k: v for k, v in (attrs or {}).items()}


def span_to_event(span, pid, tid):
    start_us = iso_to_us(span["start_time"])
    end_us = iso_to_us(span["end_time"])
    if start_us is None or end_us is None:
        return None
    dur = max(end_us - start_us, 0)
    args = flatten_attrs(span.get("attributes"))
    args["span_id"] = span["context"]["span_id"]
    args["trace_id"] = span["context"]["trace_id"]
    if span.get("parent_id"):
        args["parent_id"] = span["parent_id"]
    status = span.get("status", {}).get("status_code")
    if status:
        args["status"] = status
    return {
        "name": span["name"],
        "cat": "span",
        "ph": "X",
        "ts": start_us,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": args,
    }


def metric_points_to_events(metric_record, default_rank, mode, ctx):
    events = []
    for rm in metric_record.get("resource_metrics", []):
        resource_attrs = rm.get("resource", {}).get("attributes", {}) or {}
        # counters attach to the MAIN trainer process lane for this rank
        pid, _t, _p, _l = lane_for(resource_attrs, "megatron.metric", "trainer", default_rank, mode, ctx)
        for sm in rm.get("scope_metrics", []):
            for metric in sm.get("metrics", []):
                name = metric["name"]
                data = metric.get("data", {})
                for dp in data.get("data_points", []):
                    t_ns = dp.get("time_unix_nano")
                    if t_ns is None:
                        continue
                    ts_us = t_ns / 1000.0
                    if "value" in dp and dp["value"] is not None:
                        value = dp["value"]
                    elif "sum" in dp and dp["sum"] is not None:
                        # histogram: report the sum (single sample per step in
                        # this data, so sum == the observed value)
                        value = dp["sum"]
                    else:
                        continue
                    events.append({
                        "name": name,
                        "cat": "metric",
                        "ph": "C",
                        "ts": ts_us,
                        "pid": pid,
                        "args": {name: value},
                    })
    return events


# The reckoner emits an INTERVAL GRAPH, not a tree: allocation/batch/extern/step records
# overlap and outlive each other (a .extern can end after its allocation). So: each requeue
# ATTEMPT (SLUID) is ONE process (pid) -- isolating requeues whose cleanup tails can overlap
# the next attempt -- and each record type / phases / synthesized is a separate TRACK (tid)
# within it, so overlapping records sit on their own lanes instead of fake-nesting. The thin
# legacy-compat spans (slurm.job/step/... kept only for goodput) are dropped from the timeline.
_SLURM_TID = {  # record type -> lane offset within the attempt's pid
    "slurm.record.allocation": 0, "slurm.record.batch": 1, "slurm.record.extern": 2,
    "slurm.record.step": 3, "slurm.job_attempt.synthesized": 4,
}
_SLURM_LEGACY = {"slurm.job", "slurm.job_setup", "slurm.job_teardown", "slurm.step",
                 "slurm.inter_attempt_gap", "slurm.inter_step_gap"}


def slurm_track(name, attrs):
    """(pid, tid, proc_label, thread_label) for a slurm span, or None to DROP it (legacy)."""
    if name in _SLURM_LEGACY:
        return None
    attempt = int(float(attrs.get("slurm.attempt", 0) or 0))
    pid = 9000 + attempt * 10
    proc = "SLURM attempt %d (%s)" % (attempt, attrs.get("slurm.sluid", ""))
    if name.startswith("slurm.phase."):
        return pid, pid + 5, proc, "phases"
    off = _SLURM_TID.get(name)
    if off is None:
        return pid, pid + 9, proc, "other"
    # Co-scheduled INFRA steps (e.g. the per-node OTel collector on a `srun --overlap` step) get
    # their OWN lane -- they overlap/outlive the workload step, so sharing the workload step lane
    # would fake-nest. Reckoner tags these slurm.record.step with slurm.step.role=infra.
    if name == "slurm.record.step" and attrs.get("slurm.step.role") == "infra":
        return pid, pid + 7, proc, "record.step.infra"
    return pid, pid + off, proc, name.split(".", 1)[1]   # e.g. "record.allocation"


def impute_slurm_from_workload(found, events, procs, threads):
    """IMPUTED 'what happened' reconstruction (confidence=inferred), on ONE process / ONE lane:
    a `slurm.imputed.job` parent per attempt with sequential children that NEST cleanly:

        [ slurm.imputed.job                                                        ]
        [deferred][queued][setup][step .................][teardown]

    Boundaries cross SLURM facts (submit/eligible/start/step_start/max row End) with the
    training's LAST telemetry within the attempt window (process-alive proxy -- the workload
    uber span doesn't export on a kill). step = step_start->last_telemetry (productive);
    teardown = last_telemetry->max_end (process death + NCCL/CG, nodes held but idle)."""
    slurm_files = [p for p, r, _ in found if r == "slurm"]
    trainer_files = [p for p, r, _ in found if r == "trainer"]
    if not slurm_files or not trainer_files:
        return
    from collections import defaultdict
    by_att = defaultdict(dict)
    res = {}
    for o in (o for f in slurm_files for o in read_spans(f)):
        a = o.get("attributes", {}) or {}
        res = res or (o.get("resource", {}).get("attributes", {}) or {})
        att = int(float(a.get("slurm.attempt", 0) or 0))
        s, e = ts(o["start_time"]), ts(o["end_time"])
        d = by_att[att]
        d["sluid"] = a.get("slurm.sluid", "")
        d["max_end"] = max(d.get("max_end", 0.0), e)
        nm = o["name"]
        for key, match in (("step", "slurm.record.step"), ("alloc", "slurm.record.allocation"),
                           ("queued", "slurm.phase.queued"), ("deferred", "slurm.phase.deferred")):
            if nm == match:
                d[key] = (s, e)
    r0_iv = [(ts(o["start_time"]), ts(o["end_time"]))
             for o in read_spans(trainer_files[0]) if "start_time" in o and "end_time" in o]

    def add(nm, s, e, pid, tid, d):
        if s is not None and e is not None and e > s:
            events.append({"name": nm, "ph": "X", "pid": pid, "tid": tid,
                           "ts": iso_us(s), "dur": iso_us(e) - iso_us(s),
                           "args": {"confidence": "inferred", "slurm.sluid": d.get("sluid", "")}})

    for att, d in sorted(by_att.items()):
        alloc, step = d.get("alloc"), d.get("step")
        if not (alloc or step):
            continue
        pid = 9000 + att * 10                            # SAME process as this attempt's factual records
        tid = pid + 6                                    # a dedicated 'imputed' lane (job -> children nest)
        procs.setdefault(pid, ("SLURM attempt %d" % att, res))
        threads.setdefault((pid, tid), "imputed")
        me = d["max_end"]
        astart = (alloc or step)[0]
        ss = (step or alloc)[0]                          # step start (compute onset)
        q, dfr = d.get("queued"), d.get("deferred")
        start = q[1] if q else astart                    # allocation grant = end of queue
        job_lo = (dfr or q or (start,))[0]
        lo = min(x for x in (astart, ss, job_lo) if x is not None)
        within = [e for s, e in r0_iv if s >= lo - 5 and e <= me + 5]
        we = max(within) if within else None             # last training telemetry ~= process death
        add("slurm.imputed.job", job_lo, me, pid, tid, d)          # parent (spans everything)
        if dfr:
            add("slurm.imputed.deferred", dfr[0], dfr[1], pid, tid, d)
        if q:
            add("slurm.imputed.queued", q[0], q[1], pid, tid, d)
        add("slurm.imputed.setup", start, ss, pid, tid, d)         # alloc grant -> step launch
        if we is not None and ss is not None and we > ss:
            add("slurm.imputed.step", ss, we, pid, tid, d)         # step launch -> last telemetry
            add("slurm.imputed.teardown", we, me, pid, tid, d)     # last telemetry -> node release
        else:
            add("slurm.imputed.step", ss, me, pid, tid, d)         # no telemetry -> whole occupancy


def iso_us(t):
    """unix seconds -> microseconds (perfetto unit), matching iso_to_us."""
    return t * 1_000_000.0


# ---- process/thread lane assignment (two modes) ----------------------------------------------
# The telemetry has THREE real process kinds per node, each a distinct OS process.pid:
#   main   -- a GPU-rank trainer (megatron.* / workload.* / pre_startup)
#   ckpt   -- that rank's async checkpoint WORKER (nvrx.checkpoint.*), a separate process that runs
#             CONCURRENTLY with the trainer (its spans overlap train.iteration -- that is why they
#             must NOT share a lane; the old code collapsed them because the worker reuses the
#             rank's service.instance.id and only process.pid tells them apart)
#   nvrx   -- the node's ft_launcher agent (nvrx.restart.*), node-level, one per node
# We key lanes off process identity so concurrent processes never fake-nest.
#   --lanes process (default): PID = process, TID = thread (1 today; more when spans carry thread.id)
#   --lanes node             : PID = node,    TID = process (and real threads also become TIDs)
_ROLE_IDX = {"main": 0, "ckpt": 1, "nvrx": 2}    # role band; slot = band*1000 + GLOBAL dl.rank so
                                                 # main-rank4 and ckpt-rank0 never collide (dl.rank
                                                 # is global 0..N, not per-node 0..3).


def _role_of_span(resource_attrs, name, file_role):
    if file_role == "ftlauncher" or resource_attrs.get("nvrx.role") == "ft_launcher_agent":
        return "nvrx"
    if str(name).startswith("nvrx.checkpoint"):
        return "ckpt"          # async checkpoint worker process (distinct process.pid, same rank)
    return "main"


def _node_of(resource_attrs, fallback):
    n = resource_attrs.get("host.name") or resource_attrs.get("nvrx.node") or fallback
    return str(n).split(".")[0]   # strip .cm.cluster


def lane_for(resource_attrs, name, file_role, default_rank, mode, ctx):
    """Return (pid, tid, proc_label, thread_label) for one span under the selected lane mode.
    ctx carries stable node indices so lane numbers are deterministic and cluster by node."""
    role = _role_of_span(resource_attrs, name, file_role)
    node = _node_of(resource_attrs, "n%d" % default_rank)
    rank = resource_attrs.get("dl.rank")
    rank = int(rank) if (rank is not None and str(rank).isdigit()) else (None if role == "nvrx" else default_rank)
    os_pid = resource_attrs.get("process.pid")
    ni = ctx.setdefault("nodes", {}).setdefault(node, len(ctx.get("nodes", {})))
    slot = _ROLE_IDX[role] * 1000 + (rank or 0)            # collision-free process slot within a node
    thr = resource_attrs.get("thread.id")                  # real thread (absent today -> single lane)
    thr_off = (int(thr) % 90 + 1) if (thr is not None and str(thr).isdigit()) else 0
    who = "nvrx" if role == "nvrx" else ("rank%d %s" % (rank, role))
    if mode == "node":
        pid = 10000 + ni
        tid = slot * 100 + thr_off
        return pid, tid, "node %s" % node, (who if thr_off == 0 else "%s t%s" % (who, thr))
    # default: one PID per process (ni*10000 spacing > max slot ~2000 -> no cross-node collision)
    pid = 100000 + ni * 10000 + slot
    tid = pid * 100 + thr_off
    label = ("%s · nvrx" % node if role == "nvrx" else "%s · rank%d · %s" % (node, rank, role))
    if os_pid is not None:
        label += " (os %s)" % os_pid
    return pid, tid, label, (role if thr_off == 0 else "t%s" % thr)


def convert_file(path, default_rank, role, events, procs, threads, mode, ctx):
    objects = read_objects(path)
    # Lost-work status per iteration span, from the SAME shared detector the goodput report
    # uses -- so the two views can't disagree on which iterations were redone.
    lost = iteration_status(objects) if role in ("trainer", "other") else {}
    # Synthesize a top-level 'workload' parent for the MAIN trainer process when the real uber span
    # didn't export (crash) -- [min non-pre start, max end]. OFF by default: on a spare/multi-attempt
    # node this [min,max] bridges the standby wait and the restart gap, so it "covers area that is
    # not the workload" and fake-adopts orphaned children -> strange parentage. Opt in with
    # --synthesize-workload only when you specifically want the gap-bridging envelope.
    if role in ("trainer", "other") and ctx.get("synthesize"):
        sp = [o for o in objects if o.get("name") and "start_time" in o and "end_time" in o
              and not str(o["name"]).startswith("nvrx.checkpoint")]
        if sp and not any(o["name"] == "workload" for o in sp):
            npre = [o for o in sp if o["name"] != "pre_startup"]
            if npre:
                ra = npre[0].get("resource", {}).get("attributes", {}) or {}
                pid, tid, plabel, tlabel = lane_for(ra, "workload.synthesized", role, default_rank, mode, ctx)
                ws = min(ts(o["start_time"]) for o in npre)
                we = max(ts(o["end_time"]) for o in npre)
                events.append({"name": "workload.synthesized", "ph": "X", "pid": pid, "tid": tid,
                               "ts": iso_us(ws), "dur": iso_us(we) - iso_us(ws),
                               "args": {"confidence": "inferred"}})
                procs.setdefault(pid, (plabel, ra))
                threads.setdefault((pid, tid), tlabel)
    for obj in objects:
        if "resource_metrics" in obj:
            events.extend(metric_points_to_events(obj, default_rank, mode, ctx))
            continue
        if "context" not in obj or "start_time" not in obj:
            continue
        # Rename lost iterations (e.g. megatron.train.iteration.redone) so perfetto's
        # name-hash gives them their OWN color -- no fighting its color model.
        _st = lost.get(id(obj))
        if _st and _st != "committed":
            obj = {**obj, "name": obj["name"] + "." + _st}
        resource_attrs = obj.get("resource", {}).get("attributes", {}) or {}
        if role == "slurm":
            tr = slurm_track(obj["name"], obj.get("attributes", {}) or {})
            if tr is None:
                continue  # drop legacy-compat span from the timeline
            pid, tid, proc_label, thread_label = tr
        else:
            # trainer / ckptworker / ftlauncher / other -> keyed off REAL process identity (node +
            # process.pid + role), so the concurrent async-checkpoint worker and the nvrx agent get
            # their own lanes instead of collapsing onto -- and overlapping -- the trainer's.
            pid, tid, proc_label, thread_label = lane_for(
                resource_attrs, obj["name"], role, default_rank, mode, ctx)
        ev = span_to_event(obj, pid, tid)
        if ev is None:
            continue
        events.append(ev)
        procs.setdefault(pid, (proc_label, resource_attrs))
        threads.setdefault((pid, tid), thread_label)


_EPS_US = 2.0    # microseconds. NOT 1 ns: these are unix-EPOCH microseconds (~1.78e15), where
                 # float64's ULP is ~0.25 us, so a 1 ns nudge rounds straight back to the same value
                 # (a no-op). ~2 us is the smallest increment that survives at epoch magnitude and is
                 # still invisible on a multi-minute trace. (Rebasing ts to a trace-relative origin
                 # would let 1 ns work, but that loses wall-clock correlation with the logs/reckoner.)


def separate_equal_boundaries(events):
    """Perfetto stacks ph=X slices per (pid,tid); two slices that share an EXACT edge timestamp are
    ambiguous to the Chrome stack-importer and get flagged as overlaps. Three cases: a parent and
    its first child open on the SAME event (equal start), a child ends exactly at its parent's end,
    or adjacent siblings touch (one's start == the other's end). Nudge ONLY those equal edges apart
    by 1ns so nesting is unambiguous. GENUINE (non-equal) time overlaps -- real concurrency like the
    async checkpoint worker running alongside training -- are left untouched; a nudge can't fix a
    35-second overlap and shouldn't try. Those belong on separate lanes (see pack_lanes, TODO)."""
    from collections import defaultdict
    by_track = defaultdict(list)
    for e in events:
        if e.get("ph") == "X":
            by_track[(e["pid"], e.get("tid", 0))].append(e)
    for evs in by_track.values():
        # containment DFS pre-order: by original start asc, then longer-first, so a parent is always
        # processed (and pushed) before the children it contains.
        evs.sort(key=lambda e: (e["ts"], -(e["ts"] + e["dur"])))
        stack = []   # (orig_end, nudged_start, nudged_end) of the open ancestor chain
        for e in evs:
            os_ = e["ts"]; oe_ = os_ + e["dur"]        # ORIGINAL bounds -- used to decide structure
            while stack and stack[-1][0] <= os_:        # pop ancestors that closed before this starts
                stack.pop()
            ns_, ne_ = os_, oe_
            # Only de-collide against the innermost ancestor that TRULY contains this span (oe_ inside
            # it). A span that starts inside an ancestor but ends after it is a GENUINE crosser --
            # leave it alone (clamping would fake-nest real concurrency). Sibling edge-sharing
            # (A.end == B.start, disjoint) is fine in perfetto and is NOT touched.
            if stack and oe_ <= stack[-1][0]:
                _, pns, pne = stack[-1]                 # parent's NUDGED bounds
                if ns_ <= pns:
                    ns_ = pns + _EPS_US                 # strictly after the parent's (nudged) start
                if ne_ >= pne:
                    ne_ = pne - _EPS_US                 # strictly before the parent's (nudged) end
                if ne_ <= ns_:
                    ne_ = ns_ + _EPS_US
            e["ts"] = ns_
            e["dur"] = ne_ - ns_
            stack.append((oe_, ns_, ne_))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="files or directories of lens_json *.jsonl dumps")
    ap.add_argument("-o", "--output", default="perfetto_trace.json")
    ap.add_argument("--lanes", choices=["process", "node"], default="process",
                    help="lane model. process (default): PID=process (main/ckpt/nvrx are separate "
                         "PIDs), TID=thread. node: PID=node, TID=process (and real threads) -- fewer "
                         "top-level lanes, easier for big multi-node runs.")
    ap.add_argument("--synthesize-workload", action="store_true",
                    help="fabricate a 'workload' parent when the real one didn't export. OFF by "
                         "default: it bridges standby/restart gaps and fakes parentage.")
    args = ap.parse_args()

    # Shared discovery: (path, role, rank) for every lens file. role_of maps lens_slurm ->
    # 'slurm', *_ckptworker -> 'ckptworker', lens_rank* -> 'trainer'. 'other' -> treat as a
    # trainer lane (a bare dump with no rank in the name still renders).
    found = list(discover(args.inputs))
    if not found:
        print("No input files found", file=sys.stderr)
        sys.exit(1)

    events = []
    procs = {}     # pid -> (proc_label, resource_attrs)
    threads = {}   # (pid, tid) -> thread_label
    ctx = {"synthesize": args.synthesize_workload}   # node indices + options, shared across files

    for i, (path, role, rank) in enumerate(found):
        if role == "other":
            role = "trainer"
        default_rank = rank if rank is not None else i   # filename rank, else file index
        convert_file(path, default_rank, role, events, procs, threads, args.lanes, ctx)
        print(f"  parsed {os.path.basename(path)}: running total {len(events)} events", file=sys.stderr)

    # Imputed 'what happened' track: intersect SLURM occupancy with the workload span.
    impute_slurm_from_workload(found, events, procs, threads)

    # De-collide slices that share an EXACT edge (equal-start parent/child, child-ends-at-parent,
    # touching siblings) so perfetto's stack importer doesn't flag them. Genuine overlaps untouched.
    separate_equal_boundaries(events)

    # process metadata (one per pid) + thread metadata (one per pid,tid) so Perfetto groups
    # tracks: rank N -> {trainer, ckptworker}; SLURM attempt N -> {allocation, extern, step, ...}.
    meta_events = []
    for pid, (plabel, attrs) in sorted(procs.items()):
        svc = attrs.get("service.name", "service")
        # labels already carry node/rank/role (or "SLURM attempt N"), so don't re-append host.
        proc_name = f"{svc} {plabel}"
        meta_events.append({"name": "process_name", "ph": "M", "pid": pid, "tid": pid,
                            "args": {"name": proc_name}})
        # Ordering top->bottom (low = top): pid is already node-major / slot-minor, and SLURM
        # attempt pids (9000s) sort above the node/process lanes (10000s / 100000s).
        meta_events.append({"name": "process_sort_index", "ph": "M", "pid": pid, "tid": pid,
                            "args": {"sort_index": pid}})
    # lane order WITHIN a process (low = top). SLURM record lanes keep their curated order; node/
    # process lanes fall back to tid, which is slot-major (main ranks < ckpt < nvrx).
    _th_order = {"imputed": 0, "job_attempt.synthesized": 1, "phases": 2, "record.allocation": 3,
                 "record.batch": 4, "record.step": 5, "record.extern": 6, "record.step.infra": 7}
    for (pid, tid), tlabel in sorted(threads.items()):
        meta_events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": tid,
                            "args": {"name": tlabel}})
        meta_events.append({"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": tid,
                            "args": {"sort_index": _th_order.get(tlabel, tid)}})

    trace = {"traceEvents": meta_events + events, "displayTimeUnit": "ms"}
    with open(args.output, "w") as f:
        json.dump(trace, f)

    print(f"Wrote {len(events)} events ({len(procs)} processes) to {args.output}", file=sys.stderr)
    print("Open it at https://ui.perfetto.dev (Open trace file) or chrome://tracing", file=sys.stderr)


if __name__ == "__main__":
    main()
