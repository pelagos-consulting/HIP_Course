#!/bin/bash -l

#SBATCH --account=<account>-gpu    # your account
#SBATCH --partition=gpu            # Using the gpu partition
#SBATCH --nodes=1                  # Total number of nodes
#SBATCH --gpus-per-node=8          # The number of GPU's per node
#SBATCH --exclusive                # Use this to request all the resources on a node
#SBATCH --time=00:05:00

module swap PrgEnv-gnu PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

export MPICH_GPU_SUPPORT_ENABLED=1 # Enable GPU-aware MPI communication
export OMP_NUM_THREADS=8    # Set the number of OpenMP threads <= number of cores requested per node
export OMP_PLACES=cores     #To bind to cores 
export OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.
 
# Temporal workaround for avoiding Slingshot issues on shared nodes:
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)

# Compile the software
make clean
make

# Run a job with task placement and $BIND_OPTIONS
srun -N $SLURM_JOB_NUM_NODES -n 8 -c 8 --gpus-per-node=8 --gpus-per-task=1 ./hello_jobstep.exe | sort