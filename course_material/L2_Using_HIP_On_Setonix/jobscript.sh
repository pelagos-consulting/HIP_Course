#!/bin/bash -l

#SBATCH --account=<account>-gpu    # your account
#SBATCH --partition=gpu            # Using the gpu partition
#SBATCH --ntasks=8                 # Total number of tasks
#SBATCH --ntasks-per-node=8        # Set this for 1 mpi task per compute device
#SBATCH --gpus-per-task=1          # How many HIP compute devices to allocate to a  task
#SBATCH --gpu-bind=closest         # Bind each MPI task to the nearest GPU
#SBATCH --exclusive                # Use this to request all the resources on a node
#SBATCH --time=00:05:00

module swap PrgEnv-gnu PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

#export MPICH_GPU_SUPPORT_ENABLED=1 # Enable GPU support with MPI

export OMP_NUM_THREADS=8    #cpus-per-task is set to 8 by default
export OMP_PLACES=cores     #To bind to cores 
export OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.
 
# Temporal workaround for avoiding Slingshot issues on shared nodes:
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)

# Compile the software
make clean
make

# Run a job with task placement and $BIND_OPTIONS
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS ./hello_jobstep.exe | sort