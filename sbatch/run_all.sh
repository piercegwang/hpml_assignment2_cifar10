#!/usr/bin/env sh
set -e

for((i=1; i<8; i++)); do
    if [ $i -ne 5 ]
    then
        echo "Running C${i}.sh"
        sbatch "sbatch/C${i}.sh"
    else
        echo "Running C${i}_1.sh"
        echo "Running C${i}_2.sh"
        sbatch sbatch/C5_1.sh
        sbatch sbatch/C5_2.sh
    fi
done;

echo "Running Q3.sh"
sbatch sbatch/Q3.sh

date
