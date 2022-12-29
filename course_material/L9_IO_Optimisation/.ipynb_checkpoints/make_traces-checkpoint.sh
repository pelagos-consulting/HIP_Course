#!/bin/bash

export experiment_dir=$(pwd)

export taudir=./tau

export TAU_TRACE=1
export TRACEDIR=$taudir
export PROFILEDIR=$taudir

rm -rf $taudir
mkdir -p $taudir

# Get a trace for the calculation with synchronous IO
tau_exec -T serial -opencl ./wave2d_sync.exe -gpu
cd $taudir; echo 'y' | tau_treemerge.pl
tau_trace2json ./tau.trc ./tau.edf -chrome -ignoreatomic -o trace_sync.json

# Get a trace for the calculation with asynchronous IO
cd $experiment_dir
tau_exec -T serial -opencl ./wave2d_async.exe -gpu
cd $taudir; echo 'y' | tau_treemerge.pl
tau_trace2json ./tau.trc ./tau.edf -chrome -ignoreatomic -o trace_async.json


