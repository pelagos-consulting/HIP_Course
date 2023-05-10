#!/bin/bash
rocprof -i rocprof_counters.txt -o rocprof_counters/result-$SLURM_JOBID-$SLURM_PROCID.csv ./mat_mult_profiling_mpi.exe
