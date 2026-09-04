#!/bin/bash
# =============================================================================
# OTel launch-layer template — plain sbatch, no NVRx, no job array
# =============================================================================
# Emits the nv.dl.launch trace: what happens between the scheduler starting the
# job and the workload process starting. One collector per node; the workload is
# a separate service that measures itself.
#
#   [EDIT]      change these for your site, job and workload
#   [LAUNCH]    the launch layer. Copy verbatim.
# =============================================================================

# --- [EDIT] scheduler --------------------------------------------------------
#SBATCH --job-name=otel-minimal
#SBATCH --account=example-account
#SBATCH --partition=example-partition
#SBATCH --qos=example-qos
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/path/to/example-telemetry/experiments/slurm_logs/%x_%j.out

# --- [LAUNCH] head-node timestamps -------------------------------------------
# Must be the first executable lines: everything below is inside the interval
# they bound.
NV_LAUNCH_SCRIPT_START_TIME=$(date +%s.%N)
NV_LAUNCH_JOB_START_TIME=${SLURM_JOB_START_TIME:-$NV_LAUNCH_SCRIPT_START_TIME}

# --- [EDIT] paths ------------------------------------------------------------
TEL=/path/to/example-telemetry
export CONTAINER_IMAGE=/path/to/example-container.sqsh
export CONTAINER_MOUNTS=/path/to/shared:/path/to/shared,/path/to/scratch:/path/to/scratch
RUN_DIR=${TEL}/experiments/otel_minimal/${SLURM_JOB_ID}

export OTELCOL_BIN=${TEL}/otelcol/otelcol-contrib
export OTELCOL_CFG=${SLURM_SUBMIT_DIR}/otelcol-minimal.yaml
export WORKLOAD=${SLURM_SUBMIT_DIR}/mock_workload.py

# --- [LAUNCH] OTLP -----------------------------------------------------------
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_INSECURE=true
export OTEL_EXPORTER_OTLP_TIMEOUT=5000
export OTEL_BSP_SCHEDULE_DELAY=200
export OTEL_BSP_EXPORT_TIMEOUT=5000

export OTEL_ARCHIVE_DIR=${RUN_DIR}/archive
mkdir -p "$OTEL_ARCHIVE_DIR"

# --- [LAUNCH] sluid ----------------------------------------------------------
# One sacct call serves plain jobs and array elements alike. On a plain job both
# selectors hit the same row, so sluid == array_sluid.
SACCT_BASE=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
RANK0_REF=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_0}
RANK0_REF=${RANK0_REF:-$SLURM_JOB_ID}
SLUID_ROWS=$(sacct -j "$SACCT_BASE" -X -n -P -o JobID,JobIDRaw,SLUID)
LENS_SLURM_SLUID=$(awk -F'|' -v id="$SLURM_JOB_ID" '$2==id{print $3}' <<<"$SLUID_ROWS")
LENS_SLURM_ARRAY_SLUID=$(awk -F'|' -v id="$RANK0_REF" '$1==id{print $3}' <<<"$SLUID_ROWS")

export NV_LAUNCH_JOB_START_TIME NV_LAUNCH_SCRIPT_START_TIME
export LENS_SLURM_SLUID LENS_SLURM_ARRAY_SLUID
NV_LAUNCH_RESOURCE_EXPORT=$(nemo-lens set-slurm-resource-attrs --stage sbatch) || exit $?
eval "$NV_LAUNCH_RESOURCE_EXPORT" || exit $?
echo "[otel] sluid=${LENS_SLURM_SLUID} array_sluid=${LENS_SLURM_ARRAY_SLUID}"
echo "[otel] base_resource_attributes=$OTEL_RESOURCE_ATTRIBUTES"

# --- [LAUNCH] pre-srun stamp -------------------------------------------------
# Immediately before srun: this and the in-container stamp bracket container_load.
export NV_LAUNCH_PRESRUN_TIME=$(date +%s.%N)

# --- [EDIT] srun -------------------------------------------------------------
srun --ntasks-per-node=1 \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --no-container-mount-home \
     --output="${RUN_DIR}/task-%N.log" \
     bash -c '
        # --- [LAUNCH] in-container, per node ------------------------------
        # First action: container_load ends here, so anything before it inflates
        # the span.
        export NV_LAUNCH_TASK_START_TIME=$(date +%s.%N)

        # The srun stage appends what only it can resolve. topology.* is PER TASK:
        # a preamble export would stamp one node address onto every rank.
        NV_LAUNCH_RESOURCE_EXPORT=$(nemo-lens set-slurm-resource-attrs --stage task --container-image "$CONTAINER_IMAGE") || exit $?
        eval "$NV_LAUNCH_RESOURCE_EXPORT" || exit $?
        echo "[otel] task_resource_attributes=$OTEL_RESOURCE_ATTRIBUTES"

        "$OTELCOL_BIN" --config "$OTELCOL_CFG" &
        COL=$!
        # The poll interval is the FLOOR of collector_start: at sleep 1, two nodes
        # reported 2.007s and 2.008s, which is quantisation rather than startup.
        until (exec 3<>/dev/tcp/127.0.0.1/13133) 2>/dev/null; do sleep 0.1; done
        export NV_LAUNCH_COLLECTOR_READY_TIME=$(date +%s.%N)

        export NV_LAUNCH_EXEC_TIME=$(date +%s.%N)

        # --- [LAUNCH] emit the launch trace -------------------------------
        # Backgrounded so it never delays the workload. Span names and parents
        # live here, not in the tool — the tool knows no schema.
        nemo-lens emit-spans --service nv.dl.launch \
            --span nv.dl.launch,${NV_LAUNCH_JOB_START_TIME:-$NV_LAUNCH_SCRIPT_START_TIME},$NV_LAUNCH_EXEC_TIME \
            --span nv.dl.launch.scheduler_delay,$NV_LAUNCH_JOB_START_TIME,$NV_LAUNCH_SCRIPT_START_TIME,nv.dl.launch \
            --span nv.dl.launch.sbatch_script,$NV_LAUNCH_SCRIPT_START_TIME,$NV_LAUNCH_PRESRUN_TIME,nv.dl.launch \
            --span nv.dl.launch.container_load,$NV_LAUNCH_PRESRUN_TIME,$NV_LAUNCH_TASK_START_TIME,nv.dl.launch \
            --span nv.dl.launch.collector_start,$NV_LAUNCH_TASK_START_TIME,$NV_LAUNCH_COLLECTOR_READY_TIME,nv.dl.launch &

        # --- [EDIT] workload ----------------------------------------------
        python3 "$WORKLOAD"

        # --- [LAUNCH] drain -----------------------------------------------
        sleep 2
        kill -TERM "$COL"
        wait "$COL"
     '

# --- [LAUNCH] result ---------------------------------------------------------
echo "[otel] === archive ==="
for f in "$OTEL_ARCHIVE_DIR"/*.jsonl; do
    [ -e "$f" ] || continue
    echo "--- $f : $(wc -l < "$f") lines"
done
