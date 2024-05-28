#!/bin/bash -l

#SBATCH --account=courses01-gpu    # your account
#SBATCH --partition=gpu            # Using the gpu partition
#SBATCH --reservation=HIPWorkshop
#SBATCH --nodes=1                  # Total number of nodes
#SBATCH --gres=gpu:8               # The number of GPU's (and associated allocation packs) per node
#SBATCH --exclusive                # Use this to request all the resources on a node
#SBATCH --time=00:05:00

source ../env

export MPICH_GPU_SUPPORT_ENABLED=1 # Enable GPU-aware MPI communication
export OMP_NUM_THREADS=8    # Set the number of OpenMP threads per task
export OMP_PLACES=cores     # To bind OpenMP threads to cores 
export OMP_PROC_BIND=close  # To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.
 
# Temporal workaround for avoiding Slingshot issues on shared nodes:
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)

# Compile the software
build hello_jobstep.exe

# Run a job with task placement and $BIND_OPTIONS
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=8 --cpus-per-task=8\
	--gres=gpu:8 --gpus-per-task=1 --gpu-bind=closest\
	hello_jobstep.exe | sort 
