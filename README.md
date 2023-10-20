

# HPML Assignment 2: ResNet18 and CIFAR10


## Instructions for execution of code

Code in `lab2.py` takes arguments to customize optimizers, workers, and other helpful arguments for the model and execution. In order to run the code for each exercise, use the following (with the appropriate `C1` through `C7`) command **from the root directory**:

    sbatch sbatch/C1.sh

Also note, C5 has two files `C5_1.sh` and `C5_2.sh` for the GPU and CPU-only nodes.

For running all of the code from `C1` to `C7`, use the following:

    bash sbatch/run_all.sh


## Structure of Repository

    .
    ├── README.md
    ├── lab2.py
    ├── models/
    │   ├── __init__.py
    │   ├── resnet_no_batchnorm.py
    │   └── resnet.py
    └── sbatch/
        ├── C1.sh
        ├── C2.sh
        ├── C3.sh
        ├── C4.sh
        ├── C5_1.sh
        ├── C5_2.sh
        ├── C6.sh
        ├── C7.sh
        ├── Q3.sh
        ├── Q4.sh
        └── run_all.sh

