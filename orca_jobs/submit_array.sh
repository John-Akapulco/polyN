#!/bin/sh
#SBATCH --job-name=polyN_orca
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --time=96:00:00

# Submit with: sbatch --array=1-N[%K] submit_array.sh
# where N is the number of lines in jobs_list.txt and %K caps how many
# array tasks run concurrently (pick K to fit the currently idle nodes).
# Each array task = one ORCA job (8 cores) running in its own directory
# from jobs_list.txt, line SLURM_ARRAY_TASK_ID.

module load gnu12/12.3.0
module load openmpi4/4.1.6
export PATH=/opt/ohpc/pub/software/orca_6_1_0_linux_x86-64_shared_openmpi418:$PATH
ORCA_BIN=/opt/ohpc/pub/software/orca_6_1_0_linux_x86-64_shared_openmpi418/orca

JOB_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs_list.txt)
if [ -z "$JOB_DIR" ]; then
    echo "No job dir for array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

cd "$JOB_DIR" || exit 1
NAME=$(basename "$JOB_DIR")

time "$ORCA_BIN" "$(pwd)/${NAME}.inp" > "$(pwd)/${NAME}.out"
