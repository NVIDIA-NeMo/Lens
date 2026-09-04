#!/bin/bash
#SBATCH --job-name=otel-nvrx-array-spare
#SBATCH --account=example-account
#SBATCH --partition=example-partition
#SBATCH --qos=example-qos
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --array=0-3%3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/path/to/example-telemetry/experiments/slurm_logs/%x_%j.out

NV_LAUNCH_SCRIPT_START_TIME=$(date +%s.%N)
NV_LAUNCH_JOB_START_TIME=${SLURM_JOB_START_TIME:-$NV_LAUNCH_SCRIPT_START_TIME}

TEL=/path/to/example-telemetry
export OTELCOL_BIN=${TEL}/otelcol/otelcol-contrib
export OTELCOL_CFG=${SLURM_SUBMIT_DIR}/otelcol-minimal.yaml
export CONTAINER_IMAGE=/path/to/example-container.sqsh
export CONTAINER_MOUNTS=/path/to/shared:/path/to/shared,/path/to/scratch:/path/to/scratch

export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_INSECURE=true
export OTEL_EXPORTER_OTLP_TIMEOUT=5000
export OTEL_BSP_SCHEDULE_DELAY=200
export OTEL_BSP_EXPORT_TIMEOUT=5000

RUN_DIR=${TEL}/experiments/otel_nvrx_array_spare/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export OTEL_ARCHIVE_DIR=${RUN_DIR}/archive
mkdir -p "$OTEL_ARCHIVE_DIR"

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

echo "[otel] job_start=$NV_LAUNCH_JOB_START_TIME script_start=$NV_LAUNCH_SCRIPT_START_TIME"
echo "[otel] sluid=$LENS_SLURM_SLUID array_sluid=$LENS_SLURM_ARRAY_SLUID"
echo "[otel] slurmd_node=${SLURMD_NODENAME:-unset} submit_host=${SLURM_SUBMIT_HOST:-unset}"
echo "[otel] array_task_count=${SLURM_ARRAY_TASK_COUNT} array_task_max=${SLURM_ARRAY_TASK_MAX} throttle_check"
echo "[otel] array base=$SACCT_BASE task=${SLURM_ARRAY_TASK_ID:-0} count=${SLURM_ARRAY_TASK_COUNT:-1}"
echo "[otel] base_resource_attributes=$OTEL_RESOURCE_ATTRIBUTES"

CTL=${TEL}/experiments/otel_nvrx_array_spare/ctl-${SACCT_BASE}
mkdir -p "$CTL"
RDZV_HOST_FILE=${CTL}/rdzv_host
RDZV_CLOSED=${CTL}/rdzv_closed

cancel_generation() {
    [ -n "${SLURM_ARRAY_JOB_ID:-}" ] || return 0
    echo "[otel] task 0 complete; cancelling remaining array tasks for ${SLURM_ARRAY_JOB_ID}"
    scancel "$SLURM_ARRAY_JOB_ID" || true
}

mark_generation_closed() {
    rc=$?
    echo "[otel] task 0 exiting rc=${rc}; marking rendezvous closed"
    touch "$RDZV_CLOSED"
    cancel_generation
    exit "$rc"
}

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    trap mark_generation_closed EXIT
    scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1 > "$RDZV_HOST_FILE"
fi

until [ -s "$RDZV_HOST_FILE" ]; do
    if [ -f "$RDZV_CLOSED" ]; then
        echo "[otel] rendezvous already closed before host publication; exiting task ${SLURM_ARRAY_TASK_ID}"
        exit 0
    fi
    sleep 0.5
done
if [ -f "$RDZV_CLOSED" ] && [ "$SLURM_ARRAY_TASK_ID" != "0" ]; then
    echo "[otel] rendezvous closed; exiting spare task ${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
RDZV_HOST=$(cat "$RDZV_HOST_FILE")
NNODES_SPEC=2:3
RDZV_PORT=${RDZV_PORT:-29500}
NVRX_DIR=${TEL}/experiments/nvrx/${SACCT_BASE}
mkdir -p "$NVRX_DIR"/cycle_infos
touch "$NVRX_DIR/latest_checkpointed_iteration.txt"

export WORKLOAD="ft_launcher \
    --nnodes ${NNODES_SPEC} --nproc-per-node 4 \
    --rdzv-id ${SACCT_BASE} --rdzv-endpoint ${RDZV_HOST}:${RDZV_PORT} \
    --rdzv-conf join_timeout=1200,store_connect_wait_seconds=600 \
    --max-restarts 0 \
    --ft-max-no-progress-cycles 0 \
    --ft-segment=1 \
    --ft-cycle-info-dir=${NVRX_DIR}/cycle_infos \
    --ft-checkpoint-iteration-file=${NVRX_DIR}/latest_checkpointed_iteration.txt \
    --ft-rank-section-timeouts setup:1800,step:120,checkpointing:300 \
    --ft-rank-out-of-section-timeout 300 \
    ${SLURM_SUBMIT_DIR}/mock_workload.py"

export NV_LAUNCH_PRESRUN_TIME=$(date +%s.%N)
srun -l --mpi=none --ntasks-per-node=1 \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$CONTAINER_MOUNTS" \
     --no-container-mount-home \
     --output="${RUN_DIR}/task-%N.log" \
     bash -c '
        export NV_LAUNCH_TASK_START_TIME=$(date +%s.%N)
        # The srun stage appends what only it can resolve. topology.* is PER TASK:
        # a preamble export would stamp one node address onto every rank.
        NV_LAUNCH_RESOURCE_EXPORT=$(nemo-lens set-slurm-resource-attrs --stage task --container-image "$CONTAINER_IMAGE") || exit $?
        eval "$NV_LAUNCH_RESOURCE_EXPORT" || exit $?
        echo "[otel] task_resource_attributes=$OTEL_RESOURCE_ATTRIBUTES"
        echo "[otel] $SLURMD_NODENAME presrun=$NV_LAUNCH_PRESRUN_TIME task_start=$NV_LAUNCH_TASK_START_TIME"

        "$OTELCOL_BIN" --config "$OTELCOL_CFG" &
        COL=$!
        until (exec 3<>/dev/tcp/127.0.0.1/13133) 2>/dev/null; do sleep 0.1; done
        export NV_LAUNCH_COLLECTOR_READY_TIME=$(date +%s.%N)
        echo "[otel] $SLURMD_NODENAME collector_ready=$NV_LAUNCH_COLLECTOR_READY_TIME"

        export NV_LAUNCH_EXEC_TIME=$(date +%s.%N)
        echo "[otel] $SLURMD_NODENAME exec=$NV_LAUNCH_EXEC_TIME"

        nemo-lens emit-spans --service nv.dl.launch \
            --span nv.dl.launch,${NV_LAUNCH_JOB_START_TIME:-$NV_LAUNCH_SCRIPT_START_TIME},$NV_LAUNCH_EXEC_TIME \
            --span nv.dl.launch.scheduler_delay,$NV_LAUNCH_JOB_START_TIME,$NV_LAUNCH_SCRIPT_START_TIME,nv.dl.launch \
            --span nv.dl.launch.sbatch_script,$NV_LAUNCH_SCRIPT_START_TIME,$NV_LAUNCH_PRESRUN_TIME,nv.dl.launch \
            --span nv.dl.launch.container_load,$NV_LAUNCH_PRESRUN_TIME,$NV_LAUNCH_TASK_START_TIME,nv.dl.launch \
            --span nv.dl.launch.collector_start,$NV_LAUNCH_TASK_START_TIME,$NV_LAUNCH_COLLECTOR_READY_TIME,nv.dl.launch &

        echo "[otel] $SLURMD_NODENAME workload: [$WORKLOAD]"
        echo "[otel] ft_launcher: $(command -v ft_launcher || echo NOT-ON-PATH)"
        $WORKLOAD
        RC=$?
        sleep 2
        kill -TERM "$COL" 2>/dev/null || true
        wait "$COL" 2>/dev/null || true
        exit "$RC"
     '

echo "[otel] === archive ==="
for f in "$OTEL_ARCHIVE_DIR"/*.jsonl; do
    [ -e "$f" ] || continue
    echo "--- $f : $(wc -l < "$f") lines"
    head -c 2000 "$f"; echo
done
