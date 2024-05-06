#!/bin/bash

source ../env

export experiment_dir=$(pwd)/rocprof_trace

mkdir -p $experiment_dir

# Run rocprof to make traces
rocprof --hip-trace --hsa-trace -o $experiment_dir/wave2d_sync.csv wave2d_sync.exe
rocprof --hip-trace --hsa-trace -o $experiment_dir/wave2d_async_streams.csv wave2d_async_streams.exe
rocprof --hip-trace --hsa-trace -o $experiment_dir/wave2d_async_events.csv wave2d_async_events.exe


