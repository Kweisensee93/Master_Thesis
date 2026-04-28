#!/bin/bash
#SBATCH --job-name=04_run_DLModels
#SBATCH --output=../logs/04_run_DLModels_%j.out
#SBATCH --error=../logs/04_run_DLModels_%j.err
#SBATCH --time=00:30:00   # Even with training it was under 5 minutes. Take 10 to be safe. (for 174 images)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2   # Max usage is around 60% - Hence 2 CPUs to be safe
#SBATCH --mem=4000M     # Max usage is around 1.1 G (per image)
#SBATCH --partition=epyc2
#SBATCH --account=gratis
#SBATCH --wckey=noop

START_TIME=$(date +%s)
echo "Job started at: $(date)"

# Background monitor: logs CPU & RAM every 10 seconds
MONITOR_LOG=$(mktemp /tmp/monitor_XXXX.log)
(
  while true; do
    # CPU% and RSS (KB) of all processes in this job's cgroup
    ps -u $USER -o pid,pcpu,rss --no-headers \
      | awk '{cpu+=$2; ram+=$3} END {
          printf "%s CPU=%.1f%% RAM=%.1f MB\n", strftime("%H:%M:%S"), cpu, ram/1024
        }'
    sleep 10
  done
) >> "$MONITOR_LOG" &
MONITOR_PID=$!

# Main job command
source $(conda info --base)/etc/profile.d/conda.sh

conda activate master_thesis

python \
    /storage/homefs/kw23y068/Master_Thesis/scripts/DL_models.py \
    /storage/homefs/kw23y068/Master_Thesis/config.yaml

EXIT_CODE=$?

# Stop monitor and print log
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

# --- Compute runtime ---
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$(( ELAPSED % 60 ))

echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
printf "Actual runtime:  %02d:%02d:%02d (HH:MM:SS)\n" $HOURS $MINUTES $SECONDS
echo "Per-interval CPU & RAM monitor log:"
cat "$MONITOR_LOG"

echo "SLURM accounting (sacct):"
sacct -j $SLURM_JOB_ID \
  --format=JobID,JobName,Elapsed,CPUTime,MaxRSS,AveRSS,MaxVMSize \
  --units=G

rm -f "$MONITOR_LOG"
exit $EXIT_CODE
