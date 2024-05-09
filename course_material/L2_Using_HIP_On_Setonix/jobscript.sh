#!/bin/bash -l

#SBATCH --account=<account>-gpu    # your account
#SBATCH --partition=gpu            # Using the gpu partition
#SBATCH --nodes=1                  # Total number of nodes
#SBATCH --gres=gpu:8               # The number of GPU's (and associated allocation packs) per node
#SBATCH --exclusive                # Use this to request all the resources on a node
#SBATCH --time=00:05:00

module swap PrgEnv-gnu PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

export MPICH_GPU_SUPPORT_ENABLED=1 # Enable GPU-aware MPI communication
export OMP_NUM_THREADS=8    # Set the number of OpenMP threads per task
export OMP_PLACES=cores     # To bind OpenMP threads to cores 
export OMP_PROC_BIND=close  # To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.
 
# Temporal workaround for avoiding Slingshot issues on shared nodes:
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)

# Compile the software
make clean
make

# Run a job with task placement and $BIND_OPTIONS
srun \
    --nodes=$SLURM_JOB_NUM_NODES \ # Number of nodes 
    --ntasks=8 \ # Total number of MPI tasks to request, 
    \ # should be nodes*allocation_packs/gpus_per_task  
    --cpus-per-task=8 \ # Number of cores to allocate per task, usually always 8
    --gres=gpu:8 \ # Number of GCD's (allocation packs per note)
    --gpus-per-task=1 \ # Number of GPU's per task
    --gpu-bind=closest \ # Attempt to bind each GPU to the closest chiplet
    ./hello_jobstep.exe | sort # Application with arguments