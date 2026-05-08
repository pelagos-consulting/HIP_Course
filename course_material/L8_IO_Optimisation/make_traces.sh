#!/bin/bash

source ../env

export experiment_dir=$(pwd)/rocprof_trace

mkdir -p $experiment_dir

# Run rocprof to make traces
rocprofv3 --sys-trace --output-format csv pftrace -o $experiment_dir/wave2d_sync.csv -- wave2d_sync.exe
rocprofv3 --sys-trace --output-format csv pftrace -o $experiment_dir/wave2d_async_streams.csv -- wave2d_async_streams.exe
rocprofv3 --sys-trace --output-format csv pftrace -o $experiment_dir/wave2d_async_events.csv -- wave2d_async_events.exe


