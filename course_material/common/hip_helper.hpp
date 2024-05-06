///
/// @file  hip_helper.hpp
/// 
/// @brief Helper functions for HIP.
///
/// Written by Dr. Toby Potter 
/// for the Commonwealth Scientific and Industrial Research Organisation of Australia (CSIRO).
/// https://www.pelagos-consulting.com
/// tobympotter@gmail.com

/// This file is licensed under the Creative Commons Attribution 3.0 Unported License.
/// To view a copy of this license, visit http://creativecommons.org/licenses/by/3.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

#ifndef HIP_HELPER
#define HIP_HELPER

typedef float float_type;
typedef float4 float_vec_type;

// Windows specific header instructions
#if defined(_WIN32) || defined(_WIN64)
    #define NOMINMAX
    #include <windows.h>
    #include <malloc.h>
#else
    #include <unistd.h>
#endif

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <new>
#include <cassert>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdint>

// Import the HIP header
#include <hip/hip_runtime.h>

// Time to use for faulty runs
#define FAULTY_TIME -1.0

/// Examine an error code and exit if necessary.
void h_errchk(hipError_t errcode, const char* message);

/// Macro to check error codes.
#define H_ERRCHK(cmd) \
{\
    std::string file = __FILE__;\
    std::string mesg = file + ":" + std::to_string(__LINE__);\
    h_errchk(cmd, mesg.c_str());\
}

/// Get the L1 cache line size
size_t h_get_cache_line_size();

/// Find the greatest common divisor for two numbers
size_t h_gcd(size_t a, size_t b);

/// Find the least common multiple for two numbers
size_t h_lcm(size_t a, size_t b);

/// Get a "safe" value for memory alignment
size_t h_get_alignment();

/// Check to see if device supports managed memory, exit if it does not
void h_check_managed(int dev_index);

/// Show command line options
void h_show_options(const char* name);

/// Parse command line arguments to extract device index to use
int h_parse_args(int argc, char** argv);

/// Reset primary contexts on all compute devices
void h_reset_devices(int num_devices);

/// Initialise a primary context for all HIP devices
void h_acquire_devices(int* num_devices, int default_device_id);

/// Function to report information on a compute device
void h_report_on_device(int device_id);

/// Release compute devices
void h_release_devices(int num_devices);

/// Create a number of streams
hipStream_t* h_create_streams(size_t nstreams, int synchronise);

/// Release streams that were created
void h_release_streams(size_t nstreams, hipStream_t* streams);

/// Get the IO rate in MB/s for bytes read or written
float h_get_io_rate_MBs(float elapsed_ms, size_t nbytes);

/// Get how much time elapsed between two events that were recorded
float h_get_event_time_ms(
        // Assumes start and stop events have been recorded
        // with the hipEventRecord() function
        hipEvent_t t1,
        hipEvent_t t2,
        const char* message, 
        size_t* nbytes);

/// Make grid_nblocks big enough to fit a grid of at least global_size
void h_fit_blocks(dim3* grid_nblocks, dim3 global_size, dim3 block_size);

/// Allocate aligned memory for use on the host
void* h_alloc(size_t nbytes);

/// Open the file for reading and use std::fread to read in the file
void* h_read_binary(const char* filename, size_t *nbytes);

/// Write binary data to file
void h_write_binary(void* data, const char* filename, size_t nbytes);

/// Function to run a kernel
float h_run_kernel(
    // Function address
    const void* kernel_function,
    // Arguments to the kernel function
    void** kernel_args,
    // Desired global size
    dim3 global_size,
    // Size of each block
    dim3 block_size,
    // Number of shared bytes to use
    size_t* shared_bytes,
    // Which stream to use
    hipStream_t stream,
    // 0 for an ordered kernel launch, 1 for an out of order kernel launch
    int non_ordered_launch,
    // Function to prepare the kernel for launch
    // Needs flags for prep_kernel_function
    void (*prep_kernel_function)(const void*, void**, dim3, dim3, size_t*, void**),
    void** prep_kernel_args);

/// Function to optimise the block size
/// if command line arguments are --block_file or -block_file.
/// Read an input file called input_block.dat of
/// type == uint32_t, and dimensions == (nexperiments, ndim) with row major ordering.
///
/// Writes to a file called output_block.dat of 
/// type == double and dimensions == (nexperiments, 2) with row major ordering
/// and where each line is (avg, stdev) in milliseconds
void h_optimise_block(
        int argc,
        char** argv,
        const void* kernel_function,
        void** kernel_args,
        // Desired global size of the problem
        dim3 global_size,
        // Default block_size
        dim3* default_block_size,
        // Number of times to run the kernel per experiment
        size_t nstats,
        // Any prior times we should add to the result
        float prior_times,
        // Function to prepare the kernel arguments and shared memory requirements
        void (*prep_kernel_function)(const void*, void**, dim3, dim3, size_t*, void**),
        void** prep_kernel_args);

/// Kernel function to perform a 3D memory copy
/// either between allocations on a device
/// or between host and device if the host memory
/// is pinned and mapped or managed
template <class T>
__global__ void h_copy_kernel3D(
        // Destination array
        T* dst,
        // Source array
        T* src,
        // starting offset for dst
        size_t offset_dst, 
        // starting offset for src
        size_t offset_src,
        // size of 3 fastest dimensions of dst (nelements, npencils, nplanes)
        dim3 dims_dst, 
        // size of 3 fastest dimensions of src (nelements, npencils, nplanes)
        dim3 dims_src, 
        // Size of the 3D region to copy (nelements, npencils, nplanes)
        dim3 region) {
    
    // Get the x, y, z coordinates within the grid
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ((x<region.x) && (y<region.y) && (z<region.z))  {
        // Perform the actual copy
        dst[offset_dst+x + y*dims_dst.x + z*dims_dst.x*dims_dst.y] = \
        src[offset_src+x + y*dims_src.x + z*dims_src.x*dims_src.y];
    }
}

/// Function to perform a 3D memory copy
/// either between allocations on a device
/// or between host and device if the host memory
/// is pinned and mapped or managed
template <class T>
void h_memcpy3D(
        // Can be pinned or managed host memory
        T* dst, 
        // Size of dst allocation, in units of (nelements, npencils, nplanes)
        dim3 dims_dst, 
        T* src,
        // Size of src allocation, in units of (nelements, npencils, nplanes)
        dim3 dims_src,
        // Size of the region to copy in units of (nelements, npencils, nplanes)
        dim3 region,
        // Stream to perform the copy in
        hipStream_t stream,
        // Starting offsets (in elements) for dst and src
        size_t offset_dst,
        size_t offset_src) {
    
    // Copy square regions
    dim3 block_size = {16,4,1};
    
    // Make the block size 1D
    if (region.y<=1) {
        block_size.x = 64;
        block_size.y = 1;
    }
    
    // Number of blocks in the kernel
    dim3 grid_nblocks;
    
    // Choose the number of blocks so that Grid fits within it.
    h_fit_blocks(&grid_nblocks, region, block_size);
    
    // Run the kernel to copy things
    hipLaunchKernelGGL(h_copy_kernel3D, 
            grid_nblocks, 
            block_size, 0, stream, 
            dst, src,
            offset_dst, offset_src,
            dims_dst, dims_src,
            region
    );
    
    // Check the status of the kernel launch
    H_ERRCHK(hipGetLastError());
}

#endif
