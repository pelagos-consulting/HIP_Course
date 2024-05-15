#!/bin/bash --login

#SBATCH --account=courses01-gpu    # your account
#SBATCH --partition=gpu-dev        # Using the gpu-dev partition
#SBATCH --nodes=1                  # Total number of nodes
#SBATCH --gres=gpu:2               # Number of GPU's per node
#SBATCH --time=01:00:00

echo "Hostname is $HOSTNAME"

source ../../env

export OMP_NUM_THREADS=8   #To define the number of OpenMP threads available per MPI task
export OMP_PLACES=cores     #To bind to cores 
export OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.

# Temporal workaround for avoiding Slingshot issues on shared nodes:
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)

# Make the result directory
mkdir -p rocprof_counters

# Run the profiling job
srun -N $SLURM_JOB_NUM_NODES -n 2 -c 8 --gpus-per-task=1 --gpu-bind=closest ./profile.sh
