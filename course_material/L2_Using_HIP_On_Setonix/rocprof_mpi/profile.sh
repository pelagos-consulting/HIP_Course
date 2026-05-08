#!/bin/bash

mkdir -p rocprof_counters
cd rocprof_counters

rocprofv3 -i ../rocprof_counters.txt -o result-$SLURM_JOBID-$SLURM_PROCID -- mat_mult_profiling_mpi..exe
