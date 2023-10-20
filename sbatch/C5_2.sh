#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=edu      # The account name for the job.
#SBATCH --job-name=pgw2115_hpml_c5    # The job name.
#SBATCH -c 16                     # The number of cpu cores to use.
#SBATCH --time=600:00              # The time the job will take to run
#SBATCH --mem-per-cpu=8gb        # The memory the job will use per cpu core.

module load anaconda

echo "Running script"
echo "CPU:"
python lab2.py --dlw 4 --cuda 0 2>/dev/null
date
echo "Done"

# End of script
