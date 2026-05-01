#!/bin/bash

rocprofv3 -i rocprof_counters.txt -o rocprof_counters/result-$SLURM_JOBID-$SLURM_PROCID -- mat_mult_profiling_mpi.exe
