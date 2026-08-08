#!/usr/bin/env python3
"""Post-hoc SLURM span reckoner.  Usage:  slurm_telemetry_reckoner.py <jobid> [<lens_json_dir>]

Emits the job-level nemo-lens spans a training run can't emit for itself -- the
allocation envelope (slurm.job), prolog (alloc grant -> first srun step), epilog
(last step -> alloc release), and inter-attempt gaps across requeues -- by scraping
the FINAL sacct record and backdating real lens spans to those windows.

Intended to run as a lightweight cpu-partition job held by --dependency=afterany on the
training job, so it can't start until the training has fully torn down (epilog included)
and sacct's End is final. Survives training crashes (it's a separate job), which is the
whole point: the in-process slurm.job span never closes on a hard crash, so it never
exports; this always does.

It uses the real lens SDK -- NemoLensConfig.from_env() + setup_telemetry() + the ordinary
start_span(start_time=)/end(end_time=) cycle with the same lens.group/lens.span_category
tagging training uses -- so the emitted spans land in whatever backend the run's inherited
NEMO_LENS_* / OTEL_* env points at and correlate with the training spans by slurm.sluid.
When the exporter is 'console' they are written to <lens_json_dir>/lens_slurm.jsonl.gz.

See docs/observability/slurm-telemetry-reckoner.mdx for the pattern.
"""
import os
import sys
import gzip
import socket
import subprocess
from datetime import datetime


# ---- lens SDK: honor the RUN's exporter config, not a hardcoded one ---------------
# The reckoner is submitted with sbatch's default --export=ALL, so it inherits the training
# job's NEMO_LENS_* / OTEL_* env. NemoLensConfig.from_env() therefore reproduces the same
# exporter (console/otlp/wandb/honeycomb), endpoint, and service_name the run used, so the
# reckoner's spans land in the SAME backend with matching resource metadata. We only special-
# case console exactly like training: redirect it to a gzip file in json_dir; for any real
# backend we leave span_exporter=None and let lens build it.
def setup_lens(jobid, json_dir):
    os.environ.setdefault("NEMO_LENS_EXPORT_STRATEGY", "all_ranks")
    from nemo.lens.config import NemoLensConfig
    from nemo.lens import setup_telemetry
    from opentelemetry import trace as _trace

    cfg = NemoLensConfig.from_env()
    cfg.enabled = True                       # force on regardless of how env left it
    cfg.export_strategy = "all_ranks"        # single-proc reckoner must export
    cfg.span_groups = "all"

    span_exporter, closer = None, None
    if cfg.exporter == "console" and json_dir:
        os.makedirs(json_dir, exist_ok=True)
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        jf = gzip.open(os.path.join(json_dir, "lens_slurm.jsonl.gz"), "at")
        span_exporter = ConsoleSpanExporter(out=jf, formatter=lambda s: s.to_json(indent=None) + "\n")
        closer = jf

    # Basic metadata to match the run: service.name/version come from the inherited config
    # (from_env) + lens __version__; stamp the job id at resource level so these spans join
    # the run's trace in the backend even when there's no shared json dir.
    setup_telemetry(cfg, rank=0, world_size=1, span_exporter=span_exporter,
                    resource_attributes={"slurm.job.id": str(jobid),
                                         "nemo.lens.emitter": "slurm_reckoner"})
    tracer = _trace.get_tracer("nemo.lens.slurm_reckoner")
    dest = (f"console->{os.path.join(json_dir, 'lens_slurm.jsonl.gz')}"
            if closer is not None else f"{cfg.exporter} (service={cfg.service_name})")

    def close():
        _trace.get_tracer_provider().force_flush()
        _trace.get_tracer_provider().shutdown()
        if closer is not None:
            closer.flush()
            closer.close()

    return tracer, close, dest


def _tag_span(span, group="job", default_category="goodput"):
    """lens.group (+ lens.span_category) tagging. The reckoner's SLURM envelope IS goodput
    (scheduling/allocation, what a goodput report consumes), and it runs standalone WITHOUT
    the workload's group->category map, so category_of('job') is usually None. Default the
    category to goodput so these spans hit the goodput route like training's."""
    try:
        span.set_attribute("lens.group", group)
    except Exception:  # noqa: BLE001 -- telemetry must never break
        return
    c = default_category
    try:
        from nemo.lens.state import category_of
        c = category_of(group) or default_category
    except Exception:  # noqa: BLE001
        pass
    if c is not None:
        try:
            span.set_attribute("lens.span_category", c)
        except Exception:  # noqa: BLE001
            pass


def backdated_span(tracer, name, start_epoch, end_epoch, attrs=None):
    """Closed, backdated lens span -- the ordinary start_span(start_time=)/end(end_time=)
    cycle. Inherits the current context span as parent, so calling inside an attached
    slurm.job nests it under."""
    sp = tracer.start_span(name, start_time=int(round(start_epoch * 1e9)))
    _tag_span(sp)
    for k, v in (attrs or {}).items():
        if v is not None:
            sp.set_attribute(k, v)
    sp.end(end_time=int(round(end_epoch * 1e9)))


# ---- sacct derivation ----
def _t(s):
    if not s or s in ("Unknown", "None", "N/A", ""):
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


_FIELDS = "SLUID,JobIDRaw,Submit,Eligible,Start,End,State,ExitCode,DerivedExitCode,Reason,Restarts,NNodes,NodeList,JobName"

# Steps whose JobName marks them as INFRA co-scheduled alongside the workload (e.g. a per-node
# OTel collector launched as a `srun --overlap` step) rather than the training itself. They still
# get factual slurm.record.step spans (tagged slurm.step.role=infra), but are EXCLUDED from the
# derived setup/step/teardown boundary math -- otherwise an --overlap collector that outlives the
# training step would masquerade as workload time. Extendable via env (comma-separated JobNames).
_INFRA_STEP_NAMES = {"otelcol-contrib", "otelcol"} | {
    s.strip() for s in os.environ.get("LENS_RECKON_INFRA_STEP_NAMES", "").split(",") if s.strip()}


def sacct_attempts(jobid, timeout=60):
    """Parse sacct into a FACTUAL, SLUID-keyed interval graph. One 'attempt' per SLUID -- the
    canonical Slurm unique id for a requeue episode (rows within an episode all share it, so
    grouping is exact; no time-window heuristics; supersedes DBIndex). Each attempt keeps its
    raw rows (allocation / .batch / .extern / .N steps) with Submit/Eligible/Start/End and
    metadata. NO clamping, NO containment assumptions -- rows are facts (and overlap / outlive
    each other, e.g. .extern past the allocation End on a scancel). Interpretation (which end
    'counts', cleanup vs blocking) is the analysis layer's job: reckoner factual, goodput reconstructs."""
    txt = subprocess.check_output(
        ["sacct", "-j", str(jobid), "-D", "-P", "-n", "-o", _FIELDS], text=True, timeout=timeout,
    )
    groups, order = {}, []
    for line in txt.strip().splitlines():
        if not line:
            continue
        (sluid, raw, submit, elig, start, end, state, exitc, dexit,
         reason, restarts, nnodes, nodes, jobname) = (line.split("|") + [""] * 14)[:14]
        if sluid not in groups:
            groups[sluid] = {"sluid": sluid, "rows": []}
            order.append(sluid)
        suffix = raw.rsplit(".", 1)[1] if "." in raw else ""
        rtype = ("allocation" if suffix == "" else suffix if suffix in ("batch", "extern")
                 else "step" if suffix.isdigit() else "other")
        is_infra = rtype == "step" and jobname in _INFRA_STEP_NAMES
        groups[sluid]["rows"].append({
            "raw": raw, "row_type": rtype, "jobname": jobname,
            "step_role": ("infra" if is_infra else "workload") if rtype == "step" else None,
            "submit": _t(submit), "eligible": _t(elig), "start": _t(start), "end": _t(end),
            "state": state, "exit": exitc, "dexit": dexit, "reason": reason,
            "restarts": restarts, "nnodes": nnodes, "nodelist": nodes,
        })
    attempts = [groups[k] for k in order]
    for i, a in enumerate(attempts):
        rows = a["rows"]
        a["alloc"] = next((r for r in rows if r["row_type"] == "allocation"), None)
        a["steps"] = sorted((r for r in rows if r["row_type"] == "step"),
                            key=lambda r: (r["start"] is None, r["start"]))
        # Workload steps only (drop co-scheduled infra like the --overlap collector) -- the derived
        # setup/step/teardown decomposition keys off these so infra can't masquerade as workload.
        a["workload_steps"] = [r for r in a["steps"] if r["step_role"] != "infra"]
        a["attempt_index"] = i
        ends = [r["end"] for r in rows if r["end"] is not None]
        a["max_end"] = max(ends) if ends else None
        subs = [r["submit"] for r in rows if r["submit"] is not None]
        a["submit"] = (a["alloc"]["submit"] if a["alloc"] and a["alloc"]["submit"] is not None
                       else (min(subs) if subs else None))
    attempts.sort(key=lambda a: (a["alloc"] is None or a["alloc"]["start"] is None,
                                 a["alloc"]["start"] if (a["alloc"] and a["alloc"]["start"]) else 0.0))
    return attempts


def emit_slurm_spans(tracer, attempts, jobid):
    """Emit a FLAT interval graph -- no context nesting, because sacct rows overlap and don't
    nest (a 'child' .extern can outlive the 'parent' allocation). Per attempt:
      * slurm.job_attempt.synthesized  Submit -> max(row End)   (grouping only; confidence=synthesized)
      * slurm.phase.deferred           Submit -> Eligible       (hold/backoff, e.g. BeginTime)
      * slurm.phase.queued             Eligible -> Start        (real scheduler/resource wait)
      * slurm.record.{allocation,batch,extern,step}  raw row Start->End   (factual, overlapping)
      * slurm.phase.cleanup_tail       alloc End -> max(row End) (may be ONE node in CG; blocking
                                                                  semantics are the analysis's call)
    Plus a THIN legacy set (slurm.job/job_setup/step/job_teardown/inter_attempt_gap) derived
    honestly (job window = alloc Start -> max row End, so no clamp) so a current goodput report
    keeps working during migration to the factual model."""
    def emit(name, s, e, attrs):
        if s is not None and e is not None and e > s:
            backdated_span(tracer, name, s, e, attrs)

    prev_end = None
    for a in attempts:
        alloc = a["alloc"] or {}
        idx, me, submit = a["attempt_index"], a["max_end"], a["submit"]
        elig, start, aend = alloc.get("eligible"), alloc.get("start"), alloc.get("end")
        prov = {"slurm.job.id": str(jobid), "slurm.sluid": a["sluid"],
                "slurm.attempt": idx, "slurm.source": "sacct", "slurm.confidence": "factual",
                "slurm.state": alloc.get("state", ""), "slurm.reason": alloc.get("reason", ""),
                "slurm.restarts": alloc.get("restarts", ""), "slurm.nodelist": alloc.get("nodelist", "")}
        # --- factual model ---
        emit("slurm.job_attempt.synthesized", submit, me, {**prov, "slurm.confidence": "synthesized"})
        emit("slurm.phase.deferred", submit, elig, prov)
        emit("slurm.phase.queued", elig, start, prov)
        for r in a["rows"]:
            step_attrs = ({"slurm.step.name": r["jobname"], "slurm.step.role": r["step_role"]}
                          if r["row_type"] == "step" else {})
            emit(f"slurm.record.{r['row_type']}", r["start"], r["end"],
                 {**prov, "slurm.row_type": r["row_type"], "slurm.state": r["state"],
                  "slurm.exit_code": r["exit"], "slurm.nodelist": r["nodelist"], **step_attrs})
        emit("slurm.phase.cleanup_tail", aend, me,
             {**prov, "slurm.note": "alloc End -> max row End; may be one node stuck in CG (inferred)"})

        # --- thin legacy compat (honest window to max_end, no clamp) ---
        # WORKLOAD steps only: an --overlap infra step (collector) must not define the workload
        # setup/step/teardown boundaries (it outlives the training step -> would inflate 'step').
        steps = a["workload_steps"]
        s_first = steps[0]["start"] if steps else start
        s_last = steps[-1]["end"] if steps else aend
        j0, j1 = start, me
        lg = {"slurm.job.id": str(jobid), "slurm.attempt": idx, "slurm.state": alloc.get("state", ""),
              "slurm.nodelist": alloc.get("nodelist", ""), "slurm.source": "reckoner"}
        emit("slurm.job", j0, j1, {**lg, "slurm.step_count": len(steps)})
        emit("slurm.job_setup", j0, s_first, lg)
        for k, r in enumerate(steps):
            emit("slurm.step", r["start"], r["end"], {**lg, "slurm.step_index": k})
            if k < len(steps) - 1:
                emit("slurm.inter_step_gap", steps[k]["end"], steps[k + 1]["start"],
                     {**lg, "slurm.step_gap_index": k})
        emit("slurm.job_teardown", s_last, j1, lg)
        if prev_end is not None and j0 is not None and j0 > prev_end:
            emit("slurm.inter_attempt_gap", prev_end, j0, {**lg, "slurm.gap_before_attempt": idx})
        prev_end = j1 if j1 is not None else prev_end


def main(jobid, json_dir):
    print(f"[reckon] job={jobid}  json_dir={json_dir}  host={socket.gethostname()}")
    try:
        attempts = sacct_attempts(jobid)
    except Exception as e:  # noqa: BLE001
        print(f"[reckon] sacct FAILED -> {e}")
        return 1
    if not attempts:
        print("[reckon] no sacct records; nothing to emit")
        return 1
    def _d(x, y):
        return None if (x is None or y is None) else round(y - x, 1)
    for a in attempts:
        al = a["alloc"] or {}
        wsteps = a["workload_steps"]
        s0 = wsteps[0]["start"] if wsteps else al.get("start")
        s1 = wsteps[-1]["end"] if wsteps else al.get("end")
        infra = [s["jobname"] for s in a["steps"] if s["step_role"] == "infra"]
        print(f"[reckon] attempt {a['attempt_index']} sluid={a['sluid']} state={al.get('state','?')} "
              f"steps={len(a['steps'])} (workload={len(wsteps)}, infra={infra or '-'}) nodes={al.get('nodelist','')}")
        print(f"        deferred={_d(a['submit'], al.get('eligible'))}s "
              f"queued={_d(al.get('eligible'), al.get('start'))}s "
              f"setup={_d(al.get('start'), s0)}s step={_d(s0, s1)}s "
              f"cleanup_tail={_d(al.get('end'), a['max_end'])}s")
    tracer, close, dest = setup_lens(jobid, json_dir)
    emit_slurm_spans(tracer, attempts, jobid)
    close()
    print(f"[reckon] emitted SLURM spans -> {dest}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: slurm_telemetry_reckoner.py <jobid> [<lens_json_dir>]", file=sys.stderr)
        print("  json_dir is only used when the run's exporter is 'console' (file redirect);", file=sys.stderr)
        print("  for otlp/wandb/honeycomb the reckoner inherits the config from env and ships there.", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
