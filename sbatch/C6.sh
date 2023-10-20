#!/bin/sh
#
# Simple "Hello World" submit script for Slurm.
#
# Replace <ACCOUNT> with your account name before submitting.
#
#SBATCH --account=edu      # The account name for the job.
#SBATCH --job-name=pgw2115_hpml_c5    # The job name.
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=30:00              # The time the job will take to run
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.
#SBATCH --gres=gpu

module load cuda11.7/toolkit
module load anaconda

echo "Running script"
echo "SGD:"
python lab2.py --dlw 4 --optimizer "sgd" 2>/dev/null
echo "SGD+Nesterov:"
python lab2.py --dlw 4 --optimizer "sgd+n" 2>/dev/null
echo "Adagrad:"
python lab2.py --dlw 4 --optimizer "adagrad" 2>/dev/null
echo "Adadelta:"
python lab2.py --dlw 4 --optimizer "adadelta" 2>/dev/null
echo "Adam:"
python lab2.py --dlw 4 --optimizer "adam" 2>/dev/null
date
echo "Done"

# End of script
