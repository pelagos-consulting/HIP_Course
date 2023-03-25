#!/bin/bash -l

#SBATCH --account=pawsey0007       # your account
#SBATCH --partition=gpu            # Using the gpu partition
#SBATCH --ntasks=8                 # Total number of tasks
#SBATCH --ntasks-per-node=8        # Set this for 1 mpi task per compute device
#SBATCH --cpus-per-task=16          # How many OpenMP threads per MPI task 
#SBATCH --threads-per-core=2       # How many threads per core
#SBATCH --gpus-per-task=1          # How many HIP compute devices to allocate to a  task
#SBATCH --mem=4000M                #Indicate the amount of memory per node when asking for share resources
#SBATCH --time=01:00:00

module swap PrgEnv-gnu PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   #To define the number of OpenMP threads available per MPI task, in this case it will be 8
export OMP_PLACES=cores     #To bind to cores 
export OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.
 
# Temporal workaround for avoiding Slingshot issues on shared nodes:
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)

# Bind options to srun plane=4 is the number of NUMA domains per processor (4)
export BIND_OPTIONS="--cpu-bind=threads -m block:cyclic --gpu-bind=closest"

# Run a job with task placement and $BIND_OPTIONS
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS $BIND_OPTIONS  ./hello_jobstep.exe