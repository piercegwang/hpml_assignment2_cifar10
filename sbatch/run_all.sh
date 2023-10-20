#!/usr/bin/env sh
set -e

for((i=1; i<8; i++)); do
    echo "Running C$i.sh"
    sbatch "sbatch/C$i"
done;

sbatch sbatch/C5_1.sh
sbatch sbatch/C5_2.sh

date
