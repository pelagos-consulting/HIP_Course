#!/bin/bash -login
# Use a login shell to have a clean environment

# Get the environment
source ../env

# Swap modules
module swap PrgEnv-gnu PrgEnv-cray

# Run the make command
make clean
make
