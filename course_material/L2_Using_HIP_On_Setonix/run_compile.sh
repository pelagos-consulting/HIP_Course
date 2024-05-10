#!/bin/bash -login

# Use a login shell to have a clean environment
module load rocm/5.2.3
module swap PrgEnv-gnu PrgEnv-cray
module load craype-accel-amd-gfx90a

# Run the make command
make clean
make
