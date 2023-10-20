#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=edu      # The account name for the job.
#SBATCH --job-name=pgw2115_hpml_c5    # The job name.
#SBATCH -c 5                     # The number of cpu cores to use.
#SBATCH --time=60:00              # The time the job will take to run
#SBATCH --mem-per-cpu=4gb        # The memory the job will use per cpu core.
#SBATCH --gres=gpu

module load cuda11.7/toolkit
module load anaconda

echo "Running script"
echo "GPU:"
python lab2.py --dlw 4 --cuda 1 2>/dev/null
date
echo "Done"

# End of script
