#!/usr/bin/env python3
"""Per-GPU goodput report + flame from nemo-lens spans (OTel output, not perfetto).

Model (good/bad, telescope down):
  * PEAK RATE = fastest iteration AFTER the first checkpoint (steady state,
    per-rank). Only peak-rate committed compute is GOOD:  GOOD = committed_iters
    x peak_iter_time. Every iteration's excess over peak is BAD (performance
    loss). First iteration: only its peak-aligned slice is GOOD; the rest is
    first_iteration_warmup.
  * Denominator = TRUE SLURM allocation (sacct ElapsedRaw x ranks), auto-detected
    from the job id in the dir name; falls back to the slurm.job span. Coverage
    is the UNION of accounting-span intervals, so unobserved (incl. post-shutdown
    teardown SLURM sees but spans don't) is real, not a sum-vs-union artifact.
  * Three goodput factors (Drake-style, multiply to overall = GOOD/TOTAL):
        sched_goodput   = 1 - sched   / TOTAL
        runtime_goodput = 1 - runtime / (TOTAL - sched)
        app_goodput     = GOOD / (GOOD + app)
        overall         = sched x runtime x app = GOOD / TOTAL
    sched   = slurm-side. runtime = restart + defense + lost_work + unobserved.
    app     = iteration_slowdown + first_iter_warmup + reporting + eval.
  * Per-rank GPU-hours (wall-clock is bunk). Only EXPOSED trainer time charged;
    the async checkpoint worker is hidden/overlapped, reported separately.
  * No amortization -- every number is a measured span.

Config-driven (CLASSIFY + the drill/denominator spec) so it's easy to retune.

Usage:
  python goodput_report.py <lens_json_dir>... [--html f.html] [--fold f.folded]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from lens_analysis import read_spans as _read, ts as _ts, discover, iteration_status

# ============================ CONFIG (tweak here) =============================
CLASSIFY = {
    # --- training's IN-PROCESS view (fallback; DROPPED from ranks when the reckoner is
    #     present -- see the drop set in main()). New runs name these workload.*/pre_startup;
    #     older runs used slurm.* -- both are mapped here for back-compat.
    "pre_startup":                          "BAD;scheduling;pre_startup",
    "slurm.job.prolog":                     "BAD;scheduling;pre_startup",       # old name
    "workload.startup.launch_script":       "BAD;scheduling;launch_script",
    "slurm.startup.launch_script":          "BAD;scheduling;launch_script",     # old name
    # container_load is internal to the srun step (step_start -> program_start): the only
    # pre-python phase the RANK itself measures, invisible to slurm accounting -> a rank-side
    # restart cost, kept even when the reckoner owns the rest of the slurm-side envelope.
    "workload.startup.container_load":      "BAD;runtime;restart;container_load",
    "slurm.startup.container_load":         "BAD;runtime;restart;container_load",  # old name
    # --- the post-hoc RECKONER's decomposition of the allocation (parent [job_start,job_end]):
    #   job_setup (job_start->step_start) | step (structural) | job_teardown (step_end->job_end)
    #   + inter_step_gap (multistep) + inter_attempt_gap (requeue). No prolog/epilog (dropped).
    "slurm.job_setup":                      "BAD;scheduling;job_setup",
    "slurm.job_teardown":                   "BAD;scheduling;job_teardown",
    # slurm.step (step_start->step_end) stays STRUCTURAL (not a sweep-line bucket -- it would
    # swallow the whole iterations middle, which is accounted separately). Instead its window
    # is used below to split UNCOVERED time: the uncovered part INSIDE a step is the dark
    # post-telemetry teardown -> runtime;shutdown; the rest is genuinely unobserved.
    "slurm.inter_step_gap":                 "BAD;scheduling;inter_step_gap",
    "slurm.inter_attempt_gap":              "BAD;scheduling;requeue_gap",
    # runtime / restart -- in-process, getting a rank back to productive.
    # checkpoint_load is a CHILD of model_init (load happens inside model setup),
    # so it nests under model_init; model_init's own time is the ;build remainder.
    "megatron.startup.imports":             "BAD;runtime;restart;python_init",
    "megatron.startup.arg_parse":           "BAD;runtime;restart;python_init",
    "megatron.startup.inprocess_setup":     "BAD;runtime;restart;python_init",
    "megatron.startup.in_job_setup":        "BAD;runtime;restart;python_init",
    "megatron.startup.initialize_megatron": "BAD;runtime;restart;megatron_init",
    "megatron.startup.jit_fusion_options":  "BAD;runtime;restart;megatron_init",
    "megatron.startup.model_init":          "BAD;runtime;restart;model_init;build",
    "megatron.checkpoint.load":             "BAD;runtime;restart;model_init;checkpoint_load",
    "megatron.startup.dataloader":          "BAD;runtime;restart;dataloader",
    # rest_of_startup: the startup tail AFTER model/dataloader are up -- graph capture +
    # the one-time weight-hash check + the FIRST sniff test. These are part of getting the
    # rank productive (restart cost), so they live under restart, grouped together. The
    # ONGOING train-time sniff (megatron.train.sniff_test) stays under defense.
    "megatron.startup.cuda_graph_capture":  "BAD;runtime;restart;rest_of_startup;warmup",
    "megatron.startup.weight_hash_check":   "BAD;runtime;restart;rest_of_startup;weight_hash",
    "megatron.startup.sniff_test":          "BAD;runtime;restart;rest_of_startup;sniff",
    # runtime / defense -- paid while running to stay resilient
    "megatron.train.sniff_test":            "BAD;runtime;defense;sniff",
    "megatron.checkpoint.ft_heartbeat":     "BAD;runtime;defense;ft_heartbeat",
    "megatron.checkpoint.exit_finalize":    "BAD;runtime;defense;terminate",
    # application inefficiency
    "megatron.train.iteration_report":      "BAD;app;reporting",
    "megatron.evaluate":                    "BAD;app;eval",
    # --- NVRx ft_launcher spans (PER NODE; fanned onto every rank on the node in main()). cycle/run
    #     are STRUCTURAL (the run window is already accounted by the iterations) and stay UNCLASSIFIED;
    #     fault/excluded are instant markers. Goodput taxonomy per the ops model:
    #   cold_start  = time BEFORE the first cycle (outside-srun -> first nvrx) -> SCHEDULING (slurm-level).
    #   rendezvous/health_check/worker_launch/teardown = BETWEEN cycles, pure NVRx machinery -> runtime;ORCHESTRATION.
    #   await_round = a spare holding a GPU idle for resilience               -> runtime;RESILIENCE.
    "nvrx.cold_start":                      "BAD;scheduling;cold_start",
    "nvrx.restart.rendezvous":              "BAD;runtime;orchestration;rendezvous",
    "nvrx.restart.health_check":            "BAD;runtime;orchestration;health_check",
    "nvrx.restart.worker_launch":           "BAD;runtime;orchestration;worker_launch",
    "nvrx.restart.teardown":                "BAD;runtime;orchestration;teardown",
    "nvrx.restart.await_round":             "BAD;runtime;resilience;standby",
}
CKPT_SAVE_PATH = "BAD;runtime;defense;checkpoint_save"
CKPT_SAVE_PARENT = "megatron.checkpoint.exposed_save"
CKPT_SAVE_CHILDREN = {"megatron.checkpoint.save": "dispatch_staging",
                      "megatron.checkpoint.timers_log": "skew_wait"}
CKPT_SAVE_SIBLING = {"megatron.checkpoint.save.finalize": "finalize"}
ITER = "megatron.train.iteration"
WORKER_WRITE = "nvrx.checkpoint.save.write"
# Denominator span (job start->end), first found wins: the reckoner's slurm.job (real
# allocation) if present, else the training's in-process 'workload' uber span, else pretrain.
DENOM_SPANS = ["slurm.job", "workload", "megatron.pretrain"]
# Three telescoping goodput factors (multiply to overall = GOOD/TOTAL):
SCHED_PREFIXES = ("BAD;scheduling",)
RUNTIME_PREFIXES = ("BAD;runtime", "BAD;unobserved")
# Fixed display order (siblings only, so ranks repeat per level; unranked -> by size):
ORDER = {"GOOD": 0, "BAD": 1,
         # top-level factors
         "scheduling": 0, "runtime": 1, "app": 2, "unobserved": 3,
         # scheduling children -- chronological, slurm-side only (reckoner prolog/epilog
         # bracket the step; launch_script is the bash orchestration inside the prolog gap)
         "cold_start": -1, "pre_startup": 0, "job_setup": 1, "launch_script": 2, "inter_step_gap": 3,
         "job_teardown": 4, "requeue_gap": 5,
         # runtime children
         "restart": 0, "orchestration": 1, "resilience": 2, "lost_work": 3, "defense": 4,
         # orchestration children (chronological restart machinery) + resilience child
         "rendezvous": 0, "health_check": 1, "worker_launch": 2, "teardown": 3, "standby": 0,
         # unobserved children (shutdown = dark in-step teardown; untracked = dark, no step)
         "shutdown": 0, "untracked": 1,
         # restart children -- chronological startup order (container_load = the rank's
         # own in-step container spin-up, the first thing it can measure; rest_of_startup
         # = the graph-capture / weight-hash / first-sniff tail, last before training)
         "container_load": 0, "python_init": 1, "megatron_init": 2,
         "model_init": 3, "dataloader": 4, "rest_of_startup": 5,
         # nvrx agent-side restart phases (chronological: rejoin -> health -> launch -> stop)
         "nvrx_rendezvous": 6, "nvrx_health_check": 7, "nvrx_worker_launch": 8, "nvrx_teardown": 9,
         "standby": 3,  # under defense: a hot spare held idle
         # rest_of_startup children
         "weight_hash": 0, "sniff": 1, "warmup": 2,
         # lost_work children (redone = recomputed after a requeue; uncommitted = past the
         # last surviving checkpoint) and their compute/slowdown split
         "redone": 0, "uncommitted": 1,
         "compute": 0, "slowdown": 1,
         # app children -- worst-first
         "first_iteration_warmup": 0, "first_iter_after_ckpt": 1,
         "iteration_slowdown": 2, "reporting": 3, "eval": 4}
def _order(name): return ORDER.get(name, 999)
SORT = "fixed"  # 'fixed' (ORDER), 'size' (biggest first), or 'alpha'
def _skey(kv):
    if SORT == "size":  return (0, -kv[1]["_s"], kv[0])
    if SORT == "alpha": return (0, 0.0, kv[0])
    return (_order(kv[0]), -kv[1]["_s"], kv[0])
APP_PREFIXES = ("BAD;app",)
# =============================================================================


def _dur(o): return _ts(o["end_time"]) - _ts(o["start_time"])
def _it(o):
    v = (o.get("attributes") or {}).get("megatron.iteration")
    return int(v) if v is not None else None


def analyze_rank(spans, last_ckpt, first_ckpt, job_total=None, ckpt_iters=frozenset()):
    paths = defaultdict(float)
    if not spans:
        return paths, 0.0, 0.0
    # J0/J1 anchor the span timeline (for interval clipping). The DENOMINATOR is
    # the true SLURM allocation (job_total, from sacct ElapsedRaw) when available
    # -- so post-shutdown teardown (epilog/container) lands in UNOBSERVED; else
    # fall back to the slurm.job span (SLURM_JOB_START -> telemetry shutdown).
    denom = None
    for nm in DENOM_SPANS:
        cand = [o for o in spans if o["name"] == nm]
        if cand:
            denom = cand[0]
            break
    if denom is not None:
        J0, J1 = _ts(denom["start_time"]), _ts(denom["end_time"])
    else:
        J0 = min(_ts(o["start_time"]) for o in spans)
        J1 = max(_ts(o["end_time"]) for o in spans)
    total = job_total if job_total is not None else (J1 - J0)

    # collect [start,end] intervals of every accounting span, so unobserved is
    # total - UNION(intervals) (not total - SUM, which double-counts overlaps).
    intervals = []

    def cover(o):
        a, b = max(_ts(o["start_time"]), J0), min(_ts(o["end_time"]), J1)
        if b > a:
            intervals.append((a, b))

    iters = [(o, _it(o), _dur(o)) for o in spans if o["name"] == ITER]
    iters = [(o, it, d) for o, it, d in iters if it is not None]
    peak = 0.0
    if iters:
        first_exec = min(it for _, it, _ in iters)
        steady = [d for _, it, d in iters if it > first_ckpt] or [d for _, _, d in iters]
        peak = min(steady)
        # Per-iteration lost-work status from the SHARED detector (lens_analysis.iteration_
        # status) -- ONE source of truth, so these numbers and perfetto's redone-coloring
        # can't disagree. 'committed' -> GOOD + app excess; 'redone'/'uncommitted' -> lost.
        status = iteration_status(spans)
        for (o, it, d) in iters:
            cover(o)
            st = status.get(id(o), "committed")
            if st == "committed":
                # GOOD (peak) + the app-side excess (warmup / first-after-ckpt / steady
                # slowdown; if/elif/else so each iteration's excess lands in exactly one).
                paths["GOOD"] += peak
                excess = max(0.0, d - peak)
                if it == first_exec:
                    bk = "BAD;app;first_iteration_warmup"
                elif (it - 1) in ckpt_iters:
                    bk = "BAD;app;first_iter_after_ckpt"
                else:
                    bk = "BAD;app;iteration_slowdown"
                paths[bk] += excess
            else:
                # 'redone' (recomputed by a later attempt) or 'uncommitted' (final run's
                # unsaved tail). peak = the compute to redo; excess = discarded inefficiency.
                paths[f"BAD;runtime;lost_work;{st};compute"] += peak
                paths[f"BAD;runtime;lost_work;{st};slowdown"] += max(0.0, d - peak)

    # overhead spans -> NON-OVERLAPPING partition via sweep line (innermost =
    # shortest span wins each instant). Nested checkpoint sub-spans telescope
    # (save/timers_log carve out of exposed_save -> the remainder is bookkeeping)
    # and nothing double-counts, so buckets sum to exactly the covered time.
    oseg = []
    for o in spans:
        nm = o["name"]
        bk = CLASSIFY.get(nm)
        if bk is None:
            if nm == CKPT_SAVE_PARENT:
                bk = f"{CKPT_SAVE_PATH};bookkeeping"
            elif nm in CKPT_SAVE_CHILDREN:
                bk = f"{CKPT_SAVE_PATH};{CKPT_SAVE_CHILDREN[nm]}"
            elif nm in CKPT_SAVE_SIBLING:
                bk = f"{CKPT_SAVE_PATH};{CKPT_SAVE_SIBLING[nm]}"
            else:
                continue
        a, b = max(_ts(o["start_time"]), J0), min(_ts(o["end_time"]), J1)
        if b > a:
            oseg.append((a, b, bk))
    pts = sorted(set([a for a, _, _ in oseg] + [b for _, b, _ in oseg]))
    for t0, t1 in zip(pts, pts[1:]):
        if t1 - t0 <= 1e-9:
            continue
        active = [(b - a, bk) for a, b, bk in oseg if a <= t0 + 1e-9 and b >= t1 - 1e-9]
        if active:
            paths[min(active, key=lambda x: x[0])[1]] += t1 - t0
            intervals.append((t0, t1))

    intervals.sort()
    merged = []
    for a, b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    coverage = sum(b - a for a, b in merged)
    # Split the UNCOVERED time (the union gap). It's ALL dark (no telemetry), so it stays
    # under unobserved -- but the stretches INSIDE an srun step (slurm.step window) are
    # attributable: the training's spans have closed while the process/NCCL/container/step is
    # still tearing down until SLURM stamps the step End (or a crash's kill+cleanup) ->
    # unobserved;shutdown. The rest (outside every step) is unobserved;untracked.
    gaps, cur = [], J0
    for a, b in merged:
        if a > cur:
            gaps.append((cur, a))
        cur = max(cur, b)
    if cur < J1:
        gaps.append((cur, J1))
    step_iv = [(max(_ts(o["start_time"]), J0), min(_ts(o["end_time"]), J1))
               for o in spans if o["name"] == "slurm.step"]

    def _in_step(t0, t1):
        return sum(max(0.0, min(t1, s1) - max(t0, s0)) for s0, s1 in step_iv)

    shutdown_t = sum(_in_step(a, b) for a, b in gaps)
    paths["BAD;unobserved;shutdown"] += shutdown_t
    paths["BAD;unobserved;untracked"] += max(0.0, total - coverage - shutdown_t)
    return paths, peak, total


def build_tree(paths):
    root = {"_s": 0.0, "_c": {}}
    for path, s in paths.items():
        node = root
        node["_s"] += s
        for part in path.split(";"):
            node = node["_c"].setdefault(part, {"_s": 0.0, "_c": {}})
            node["_s"] += s
    return root


H = 3600.0


def print_tree(node, total, name="TOTAL", prefix="", is_last=True, is_root=True,
               parent_s=None, lines=None):
    s = node["_s"]
    ot = 100 * s / total if total else 0
    op = 100 * s / parent_s if parent_s else 100.0
    label = name if is_root else prefix + ("\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 ") + name
    tag = "  <= peak-rate committed compute" if name == "GOOD" else ""
    lines.append(f"  {label:<44}{s/H:8.3f} gpu-h {ot:6.1f}% tot {op:6.1f}% par{tag}")
    cprefix = "" if is_root else prefix + ("    " if is_last else "\u2502   ")
    kids = sorted(node["_c"].items(), key=_skey)
    for i, (k, v) in enumerate(kids):
        print_tree(v, total, k, cprefix, i == len(kids) - 1, False, s, lines)


def tree_json(node, name, path):
    return {"name": name, "path": path, "value": round(node["_s"], 3),
            "children": [tree_json(v, k, f"{path};{k}" if path else k)
                         for k, v in sorted(node["_c"].items(), key=_skey)]}


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>goodput flame</title>
<style>
:root{--hl:#ffcf33}
body{font-family:ui-monospace,Menlo,Consolas,monospace;margin:0;padding:22px;background:#fbfbfb;color:#161616}
h1{font-size:16px;margin:0 0 3px}
#sum{font-size:12.5px;color:#444;margin:0 0 12px}
#fg{position:relative;margin-bottom:6px}
.n{position:absolute;box-sizing:border-box;border:1px solid #fff;overflow:hidden;white-space:nowrap;
   font-size:11px;line-height:23px;height:25px;cursor:pointer;padding-left:4px}
.n.hl{outline:2px solid var(--hl);outline-offset:-2px;z-index:6}
.hint{color:#8a8a8a;font-size:11px;margin:6px 0 4px}
#rep{font-size:12px;border-top:1px solid #e0e0e0;padding-top:8px}
.row{white-space:pre;cursor:pointer;border-radius:3px;padding:0 3px}
.row:hover{background:#efefef}
.row.hl{background:var(--hl)}
.sw{display:inline-block;width:10px;height:10px;margin-right:8px;vertical-align:-1px;border:1px solid #8a8a8a;border-radius:2px}
#tip{position:fixed;background:#222;color:#fff;padding:5px 9px;font-size:12px;border-radius:3px;
     pointer-events:none;display:none;z-index:9;white-space:nowrap}
#ctl{margin:0 0 10px;font-size:12px;color:#555}
#ctl button{font:inherit;margin-left:6px;padding:2px 10px;border:1px solid #bbb;background:#fff;border-radius:3px;cursor:pointer}
#ctl button.on{background:#333;color:#fff;border-color:#333}
</style></head><body>
<h1>goodput flame</h1>
<div id="sum"></div>
<div id="ctl">sort:<button data-s="fixed" class="on" onclick="setSort('fixed')">fixed</button><button data-s="size" onclick="setSort('size')">size</button><button data-s="alpha" onclick="setSort('alpha')">alpha</button></div>
<div id="fg"></div>
<div class="hint">click a box or a report row to zoom (click again / TOTAL to reset) &middot; hover to cross-highlight</div>
<div id="rep"></div>
<div id="tip"></div>
<script>
var DATA=__DATA__,SUMMARY=__SUMMARY__,H=3600,ROW=25;
document.getElementById('sum').textContent=SUMMARY;
var fg=document.getElementById('fg'),rep=document.getElementById('rep'),tip=document.getElementById('tip');
var ORDER={GOOD:0,BAD:1,scheduling:0,runtime:1,app:2,unobserved:3,restart:0,lost_work:1,defense:2,shutdown:0,untracked:1,redone:0,uncommitted:1,compute:0,slowdown:1},sortMode='fixed';
function ordv(n){return (n in ORDER)?ORDER[n]:999;}
function sortKids(a){a=(a||[]).slice();a.sort(function(x,y){
 if(sortMode==='size')return y.value-x.value;
 if(sortMode==='alpha')return x.name<y.name?-1:(x.name>y.name?1:0);
 return (ordv(x.name)-ordv(y.name))||(y.value-x.value);});return a;}
function setSort(m){sortMode=m;var b=document.querySelectorAll('#ctl button');for(var i=0;i<b.length;i++)b[i].className=(b[i].getAttribute('data-s')===m)?'on':'';render();report();}
function col(p){p=p||'';
 if(p.indexOf('GOOD')===0)return '#4a9d4a';if(p.indexOf('BAD;runtime;lost_work')===0)return '#b23b3b';
 if(p.indexOf('BAD;app')===0)return '#e39134';if(p.indexOf('BAD;scheduling')===0)return '#5a8fc0';
 if(p.indexOf('BAD;runtime;restart')===0)return '#3f74ad';if(p.indexOf('BAD;runtime;defense')===0)return '#8264ad';
 if(p.indexOf('BAD;unobserved')===0)return '#9a9a9a';if(p.indexOf('BAD')===0)return '#c76a34';return '#d0d0d0';}
var boxes={},rows={};
function reg(m,p,e){(m[p]=m[p]||[]).push(e);}
function hl(p,on){[boxes,rows].forEach(function(m){(m[p]||[]).forEach(function(e){e.classList.toggle('hl',on);});});}
function W(){return Math.max(700,document.body.clientWidth-44);}
function pad(s,n){s=''+s;while(s.length<n)s+=' ';return s;}
function lpad(s,n){s=''+s;while(s.length<n)s=' '+s;return s;}
function findNode(n,p){if((n.path||'')===p)return n;var c=n.children||[];for(var i=0;i<c.length;i++){var f=findNode(c[i],p);if(f)return f;}return null;}
var focus=DATA;
function render(){
 fg.innerHTML='';for(var k in boxes)delete boxes[k];var md=0,w=W();
 (function lay(n,depth,x,ww){if(depth>md)md=depth;
   var d=document.createElement('div');d.className='n';d.dataset.path=n.path||'';
   d.style.left=x+'px';d.style.top=(depth*ROW)+'px';d.style.width=Math.max(0,ww-1)+'px';d.style.background=col(n.path);
   var pct=(100*n.value/DATA.value).toFixed(1);if(ww>30)d.textContent=n.name+'  '+pct+'%';
   d.onmouseover=function(){hl(n.path||'',true);tip.style.display='block';
     tip.textContent=(n.path||'TOTAL')+'   '+(n.value/H).toFixed(3)+' gpu-h   '+pct+'% of total';};
   d.onmousemove=function(e){tip.style.left=(e.clientX+13)+'px';tip.style.top=(e.clientY+13)+'px';};
   d.onmouseout=function(){hl(n.path||'',false);tip.style.display='none';};
   d.onclick=function(e){e.stopPropagation();focus=(n===focus?DATA:n);render();};
   reg(boxes,n.path||'',d);fg.appendChild(d);
   var cx=x,c=sortKids(n.children);for(var i=0;i<c.length;i++){if(n.value){var cw=ww*c[i].value/n.value;lay(c[i],depth+1,cx,cw);cx+=cw;}}
 })(focus,0,0,w);
 fg.style.height=((md+1)*ROW+6)+'px';
}
function report(){
 rep.innerHTML='';for(var k in rows)delete rows[k];
 (function walk(n,pv,prefix,isLast,isRoot){
   var pct=(100*n.value/DATA.value).toFixed(1),ppar=(pv?100*n.value/pv:100).toFixed(1),gph=(n.value/H).toFixed(3);
   var label=isRoot?n.name:(prefix+(isLast?'\u2514\u2500\u2500 ':'\u251c\u2500\u2500 ')+n.name);
   var r=document.createElement('div');r.className='row';r.dataset.path=n.path||'';
   r.innerHTML='<span class="sw" style="background:'+col(n.path)+'"></span>'+
     pad(label,44)+lpad(gph,8)+' gpu-h '+lpad(pct,6)+'% tot '+lpad(ppar,6)+'% par';
   r.onmouseover=function(){hl(n.path||'',true);};
   r.onmouseout=function(){hl(n.path||'',false);};
   r.onclick=function(){var f=findNode(DATA,n.path);focus=(f&&f!==focus)?f:DATA;render();window.scrollTo({top:0,behavior:'smooth'});};
   reg(rows,n.path||'',r);rep.appendChild(r);
   var cp=isRoot?'':(prefix+(isLast?'    ':'\u2502   '));
   var c=sortKids(n.children);for(var i=0;i<c.length;i++)walk(c[i],n.value,cp,i===c.length-1,false);
 })(DATA,DATA.value,'',true,true);
}
render();report();window.onresize=render;
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--html", default=None)
    ap.add_argument("--fold", default=None)
    ap.add_argument("--sort", choices=["fixed", "size", "alpha"], default="fixed",
                    help="child order: fixed=ORDER (good->bad->sched->runtime->app->unobs), "
                         "size=biggest first, alpha")
    args = ap.parse_args()
    global SORT
    SORT = args.sort

    # Shared discovery (lens_analysis.discover): trainer ranks, their checkpoint workers,
    # and the reckoner's job-envelope file -- one place knows the run-dir layout.
    trainer, worker, slurm_files, ftlauncher_files = [], [], [], []
    for f, role, _rank in discover(args.dirs):
        if role == "trainer":
            trainer.append(f)
        elif role == "ckptworker":
            worker.append(f)
        elif role == "slurm":
            slurm_files.append(f)
        elif role == "ftlauncher":
            ftlauncher_files.append(f)
    if not trainer:
        sys.exit("no trainer span files found")

    ranks = [_read(f) for f in trainer]

    # Job-level SLURM spans from the reckoner (lens_slurm.jsonl.gz): the allocation
    # envelope + job_setup/step/teardown/gaps the in-process training can't emit. Injected
    # into EVERY rank (each GPU bears the same scheduling overhead), and the reckoner's
    # authoritative slurm.job replaces the training's in-process one so the denominator
    # window extends to the true job_end.
    slurm_spans = [o for f in slurm_files for o in _read(f)]
    slurm_envelope = None
    if slurm_spans:
        jobs = [o for o in slurm_spans if o["name"] == "slurm.job"]
        inject = [o for o in slurm_spans if o["name"] != "slurm.job"]
        if jobs:
            # one denom span spanning all attempts (requeue-safe: min start -> max end,
            # so the inter-attempt gaps sit INSIDE the denominator, as scheduling cost).
            j0 = min(_ts(o["start_time"]) for o in jobs)
            j1 = max(_ts(o["end_time"]) for o in jobs)
            slurm_envelope = j1 - j0
            denom = jobs[0] if len(jobs) == 1 else {
                "name": "slurm.job",
                "start_time": min(jobs, key=lambda o: _ts(o["start_time"]))["start_time"],
                "end_time": max(jobs, key=lambda o: _ts(o["end_time"]))["end_time"]}
            inject.append(denom)
        # When the reckoner is present it OWNS the slurm-side (prolog/job_setup/step/
        # job_teardown/epilog/gaps) authoritatively, so drop the training's back-wedged
        # approximations that would otherwise overlap it in the sweep line: its in-process
        # slurm.job, slurm.job.prolog, and slurm.startup.launch_script. Keep container_load
        # (slurm.startup.container_load) -- that's the one startup phase only the rank can
        # measure (in-step, invisible to sacct), so it stays as the rank's restart cost.
        _drop = {"workload", "pre_startup", "workload.startup.launch_script",
                 "slurm.job", "slurm.job.prolog", "slurm.startup.launch_script"}  # + old names
        for spans in ranks:
            spans[:] = [o for o in spans if o["name"] not in _drop]
            spans.extend(inject)

    # NVRx ft_launcher restart overhead is emitted PER NODE (by the node's agent). Fan each node's
    # classified restart spans onto EVERY rank on that node: all local GPUs idle together while the
    # node rejoins rendezvous / relaunches workers, so each rank bears the same cost (not a per-rank
    # split). The sweep's innermost-wins keeps it from double-counting the rank's own container_load,
    # and it fills the pre-python gap that was otherwise 'unobserved'. Match on node (host.name/nvrx.node).
    def _node_of(spans):
        for o in spans:
            ra = o.get("resource", {}).get("attributes", {}) or {}
            n = ra.get("host.name") or ra.get("nvrx.node")
            if n:
                return str(n).split(".")[0]
        return None

    nvrx_by_node = defaultdict(list)
    for o in (o for f in ftlauncher_files for o in _read(f)):
        if o.get("name") in CLASSIFY:   # only the classified restart phases; cycle/run/marks skipped
            ra = o.get("resource", {}).get("attributes", {}) or {}
            node = str(ra.get("nvrx.node") or ra.get("host.name") or "").split(".")[0]
            if node:
                nvrx_by_node[node].append(o)
    if nvrx_by_node:
        for spans in ranks:
            node = _node_of(spans)
            if node in nvrx_by_node:
                spans.extend(nvrx_by_node[node])

    last_ckpt = first_ckpt = -1
    ckpt_iters = set()
    for spans in ranks:
        for o in spans:
            if o["name"] in ("megatron.checkpoint.exposed_save", "megatron.checkpoint.save"):
                it = _it(o)
                if it is not None:
                    last_ckpt = max(last_ckpt, it)
                    first_ckpt = it if first_ckpt < 0 else min(first_ckpt, it)
                    ckpt_iters.add(it)

    # True allocation window from SLURM (sacct ElapsedRaw). Auto-detected from the
    # job id in the dir name. Falls back to the slurm.job span if sacct is
    # unavailable / the job aged out of accounting.
    import re
    import subprocess
    job_total = None
    denom_src = "slurm.job span (SLURM_JOB_START -> shutdown)"
    m = re.search(r"_(\d{6,})_", args.dirs[0].rstrip("/"))
    if m:
        try:
            out = subprocess.run(["sacct", "-j", m.group(1), "--format=ElapsedRaw",
                                  "--noheader", "-P"], capture_output=True, text=True, timeout=10)
            job_total = int(out.stdout.strip().split("\n")[0])
            denom_src = f"SLURM allocation (sacct job {m.group(1)}, ElapsedRaw={job_total}s/rank)"
        except Exception:
            job_total = None
    # The reckoner's slurm.job envelope (min start -> max end across attempts) is the most
    # authoritative denominator: it spans the whole allocation INCLUDING requeue gaps,
    # where sacct ElapsedRaw is only the last/first attempt. Prefer it when present.
    if slurm_envelope is not None:
        job_total = slurm_envelope
        denom_src = f"SLURM allocation (reckoner slurm.job envelope = {slurm_envelope:.0f}s)"

    agg = defaultdict(float)
    peaks, totals = [], []
    for spans in ranks:
        paths, peak, tot = analyze_rank(spans, last_ckpt, first_ckpt, job_total, ckpt_iters)
        for p, s in paths.items():
            agg[p] += s
        if peak:
            peaks.append(peak)
        totals.append(tot)
    worker_s = sum(_dur(o) for f in worker for o in _read(f) if o["name"] == WORKER_WRITE)

    tree = build_tree(agg)
    TOT = tree["_s"]
    good = agg.get("GOOD", 0.0)
    sched_oh = sum(v for p, v in agg.items() if p.startswith(SCHED_PREFIXES))
    runtime_oh = sum(v for p, v in agg.items() if p.startswith(RUNTIME_PREFIXES))
    app_oh = sum(v for p, v in agg.items() if p.startswith(APP_PREFIXES))
    # three telescoping factors: sched x runtime x app == GOOD/TOTAL
    sg = (TOT - sched_oh) / TOT if TOT else 0
    rem = TOT - sched_oh
    rg = (rem - runtime_oh) / rem if rem else 0
    ag = good / (good + app_oh) if (good + app_oh) else 0
    overall = good / TOT if TOT else 0

    L = []
    L.append(f"\n=== GOODPUT FLAME === {len(ranks)} GPUs | first ckpt iter {first_ckpt}, last {last_ckpt}")
    if peaks:
        L.append(f"    peak iter (fastest post-first-ckpt): {min(peaks):.2f}-{max(peaks):.2f}s "
                 f"across ranks (spread {1000*(max(peaks)-min(peaks)):.0f}ms -> barrier-gated)")
    L.append(f"    denominator = {denom_src}; only EXPOSED trainer time; worker excluded\n")
    print_tree(tree, TOT, "TOTAL", "", True, True, TOT, L)
    L.append("  " + "-" * 62)
    L.append(f"  {'worker background write':<26}{worker_s/H:8.3f} gpu-h  {100*worker_s/TOT:5.1f}% tot"
             f"  (hidden/overlapped, not charged)")
    L.append("  " + "=" * 62)
    L.append(f"\n  SCHEDULING goodput = 1 - sched/TOTAL          = {100*sg:6.2f}%   "
             f"(slurm-side = {sched_oh/H:.3f} gpu-h)")
    L.append(f"  RUNTIME    goodput = 1 - runtime/(TOTAL-sched) = {100*rg:6.2f}%   "
             f"(restart+defense+lost+unobserved = {runtime_oh/H:.3f} gpu-h)")
    L.append(f"  APP        goodput = GOOD/(GOOD+app)           = {100*ag:6.2f}%   "
             f"(slowdown+warmup+reporting+eval = {app_oh/H:.3f} gpu-h)")
    L.append(f"  {'-'*60}")
    L.append(f"  OVERALL = sched x runtime x app "
             f"= {100*sg:.2f}% x {100*rg:.2f}% x {100*ag:.2f}% = {100*sg*rg*ag:6.2f}%")
    L.append(f"  OVERALL = GOOD / TOTAL = {good/H:.3f}/{TOT/H:.3f}     "
             f"           = {100*overall:6.2f}%   "
             f"[proof: |product - GOOD/TOTAL| = {abs(sg*rg*ag-overall):.2e}]\n")
    report = "\n".join(L)
    print(report)

    if args.fold:
        with open(args.fold, "w") as fh:
            for p, s in sorted(agg.items()):
                fh.write(f"{p} {int(round(s*1000))}\n")
        print(f"  folded stacks -> {args.fold}")
    if args.html:
        summary = (f"OVERALL goodput {100*overall:.1f}%  =  sched {100*sg:.1f}% x runtime {100*rg:.1f}% "
                   f"x app {100*ag:.1f}%   |   {TOT/H:.2f} gpu-h over {len(ranks)} GPUs   |   {denom_src}")
        html = HTML.replace("__DATA__", json.dumps(tree_json(tree, "TOTAL", ""))) \
                   .replace("__SUMMARY__", json.dumps(summary))
        open(args.html, "w").write(html)
        print(f"  interactive flame -> {args.html}  (open in a browser; click to zoom)")
    print()


if __name__ == "__main__":
    main()
