#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=edu      # The account name for the job.
#SBATCH --job-name=HelloWorld    # The job name.
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=1:00              # The time the job will take to run (here, 1 min)
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.
#SBATCH --gres=gpu

module load cuda11.7/toolkit
module load anaconda
pip install --user torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
echo "Hello World"
sleep 10
date

# End of script
