#!/bin/bash
#SBATCH --job-name=run
#SBATCH --account=ls_math
#SBATCH --partition=hpc
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=144:00:00
#SBATCH --array=1-1000
#SBATCH --output=data/res_outfiles/res_%A_%a.out
#SBATCH --error=data/res_outfiles/res_%A_%a.err

# -------------------------
# Parameters
# -------------------------
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT1=$SCRIPT_DIR/01_SVI.py
SCRIPT2=$SCRIPT_DIR/01_SVI_OD.py
export BATCH_DIR=${BATCH_DIR:-data}
SEED=$SLURM_ARRAY_TASK_ID
DESIGN=${DESIGN:-ED}   # default ED
OD=${OD:-0}

mkdir -p data/res_outfiles

echo "======================================"
echo "Running on host: $(hostname)"
echo "Seed: $SEED"
echo "Design: $DESIGN"
echo "OD: $OD"
echo "Python: $(which python)"
echo "======================================"

# -------------------------
# Run
# -------------------------
if [ "$OD" -eq 1 ]; then
    uv run $SCRIPT2 \
        --n_rounds 100 \
        --start_lr_SVI 1e-03 \
        --end_lr_SVI 1e-06 \
        --num_steps_SVI 10000 \
        --seed $SEED 
else
    uv run $SCRIPT1 \
        --n_rounds 100 \
        --start_lr_SVI 1e-03 \
        --end_lr_SVI 1e-06 \
        --num_steps_SVI 10000 \
        --seed $SEED \
        --procedure $DESIGN
fi

echo "======================================"
echo "Total runtime: ${SECONDS} seconds"
echo "======================================"