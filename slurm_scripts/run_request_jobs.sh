#!/bin/bash

#SBATCH --job-name=app_request_jobs
#SBATCH --account=oz413               
#SBATCH --output=/fred/oz413/ehunt/watermark_jobs/slurm_logs/%x-%j.out
#SBATCH --partition=skylake-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8            # Request 8 CPU cores
#SBATCH --mem=16G                     # Request 16 GB of system RAM
#SBATCH --time=06:00:00               # Set a 6-hour time limit
#SBATCH --gres=gpu:1                  # Request 1 GPU

export HF_HOME="/fred/oz413/ehunt/huggingface"
# Force offline mode
export HF_HUB_CACHE=$HF_HOME
export HF_HUB_OFFLINE=1


# Load Python module
module purge
module load gcc/14.2.0
module load python/3.13.1
module load cuda/12.6.0
module load cudnn/9.5.0.50-cuda-12.6.0

# Activate your virtual environment
source /fred/oz413/ehunt/Watermark-Project/venv/bin/activate

# --- Run the Detection Script ---
echo "Starting HPC worker in background..."
cd /fred/oz413/ehunt/watermark_jobs

INPUT_DIR="/fred/oz413/ehunt/watermark_jobs/input"
OUTPUT_DIR="/fred/oz413/ehunt/watermark_jobs/output"

while true; do
    # find newest job json file
    JOB_FILE=$(ls -t ${INPUT_DIR}/job_*.json 2>/dev/null | head -n 1)

    if [ -n "$JOB_FILE" ]; then
        JOB_NAME=$(basename "$JOB_FILE")
        echo "Processing $JOB_NAME ..."
        
        # run the job
        python /fred/oz413/ehunt/Watermark-Project/UI/process_job.py "$JOB_NAME"

        # move it to archive or delete it
        mv "$JOB_FILE" "${INPUT_DIR}/processed_inputs/processed_${JOB_NAME}"
    else
        sleep 5
    fi
done

echo "Job finished."
