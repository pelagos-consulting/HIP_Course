///
/// @file  cuda_helper.hpp
/// 
/// @brief Helper functions for cuda.
///
/// Written by Dr. Toby Potter 
/// for the Commonwealth Scientific and Industrial Research Organisation of Australia (CSIRO).
///

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

// Import the cuda header
#include <cuda.h>
#include <cuda_runtime.h>

// Time to use for faulty runs
#define FAULTY_TIME -1.0

/// Examine an error code and exit if necessary - runtime API version
void h_errchk(cudaError_t errcode, const char* message) {

    if (errcode != cudaSuccess) { 
        const char* errstring = cudaGetErrorString(errcode); 
        std::fprintf( 
            stderr, 
            "Error, cuda runtime api call failed at %s, error string is: %s\n", 
            message, 
            errstring 
        ); 
        exit(EXIT_FAILURE); 
    }
}

/// Examine an error code and exit if necessary - driver API version
void h_errchk(CUresult errcode, const char* message) {

    if (errcode != CUDA_SUCCESS) { 
        char* errstring;
        CUresult status = cuGetErrorString(errcode, (const char**)&errstring); 
        
        if (status != CUDA_SUCCESS) {
        
            std::fprintf( 
                stderr, 
                "Error, cuda driver api call failed at %s, error string is: %s\n", 
                message, 
                errstring 
            );
            
        } else {
            std::fprintf( 
                stderr, 
                "Error, cuda driver api call failed at %s, error code :%d is unknown\n", 
                message, 
                errcode 
            );
        
        }
        exit(EXIT_FAILURE); 
    }
}

/// Macro to check error codes.
#define H_ERRCHK(cmd) \
{\
    std::string file = __FILE__;\
    std::string mesg = file + ":" + std::to_string(__LINE__);\
    h_errchk(cmd, mesg.c_str());\
}

/// Get the L1 cache line size
size_t h_get_cache_line_size() {
    // Get the L1 cache line size
    size_t cache_line_size = 0;

#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sys_info;
    GetSystemInfo(&sys_info);
    cache_line_size = sys_info.dwCacheLineSize;
#else
    cache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
#endif
    return cache_line_size;   
}

/// Find the greatest common divisor for two numbers
size_t h_gcd(size_t a, size_t b) {
    // Use the Euclidean algorithm to find the GCD    
    while (b != 0) {
        size_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/// Find the least common multiple for two numbers
size_t h_lcm(size_t a, size_t b) {
    size_t gcd = h_gcd(a, b);
    return (a / gcd) * b;
}

/// Get a "safe" value for memory alignment
size_t h_get_alignment() {
    // Least common multiple of L1 cache line size and 
    // Largest data type that cuda supports
    return h_lcm(h_get_cache_line_size(), sizeof(ulonglong4));
}

/// Check to see if device supports managed memory, exit if it does not
void h_check_managed(int dev_index) {
    // Check to make sure managed memory access is supported
    int managed_capability = 0;
    H_ERRCHK(
        cudaDeviceGetAttribute(
            &managed_capability,
            cudaDevAttrManagedMemory, 
            dev_index
        )
    );
    if (!managed_capability) {
        std::printf("Sorry, device %d cannot allocate managed memory\n", dev_index);
        exit(EXIT_FAILURE); 
    }    
}

/// Show command line options
void h_show_options(const char* name) {
    // Display a helpful error message
    std::printf("Usage: %s <options> <DEVICE_INDEX>\n", name);
    std::printf("Options:\n");
    std::printf("\t-h,--help\t print help message\n");
    std::printf("\tDEVICE_INDEX is a number >= 0\n"); 
}

/// Parse command line arguments to extract device index to use
int h_parse_args(int argc, char** argv) {
    
    // Default device index
    int dev_index = 0;
        
    // Check all other arguments
    for (int i=1; i<argc; i++) {
        const char* arg = (const char*)argv[i];
        
        // Check for help
        if ((std::strcmp(arg, "-h")==0) || (std::strcmp(arg, "--help")==0)) {
            h_show_options(argv[0]);
            exit(0);
        // Check for device index if it is not a flag
        } else if (std::strncmp(arg, "-", 1)!=0) {
            dev_index = (int)std::atoi(arg);
        }
    }
    
    // Make sure the index is sane
    int max_devices=0;
    H_ERRCHK(cudaGetDeviceCount(&max_devices));

    assert(dev_index<max_devices);
    assert(dev_index>=0);
    return(dev_index);
}

/// Reset primary contexts on all compute devices
void h_reset_devices(int num_devices) {
    
    // Reset devices
    for (int i = 0; i<num_devices; i++) {
        // Set device
        H_ERRCHK(cudaSetDevice(i));

        // Synchronize device 
        H_ERRCHK(cudaDeviceSynchronize());

        // Reset device (destroys primary context)
        H_ERRCHK(cudaDeviceReset());

        // Set the device to reinitialise it
        H_ERRCHK(cudaSetDevice(i));
    }
}

/// Initialise a primary context for all cuda devices
void h_acquire_devices(int* num_devices, int default_device_id) {
    
    // Initialise cuda 
    H_ERRCHK(cuInit(0));

    // Get the number of devices
    H_ERRCHK(cudaGetDeviceCount(num_devices));

    // Check to make sure we have one or more suitable devices
    if (*num_devices == 0) {
        std::printf("Failed to find a suitable compute device\n");
        exit(EXIT_FAILURE);
    }

    // Make sure the default device id is sane
    assert (default_device_id<*num_devices);

    // Clean and reset devices (optional)
    h_reset_devices(*num_devices);

    // Set the device
    H_ERRCHK(cudaSetDevice(default_device_id));
}

/// Function to report information on a compute device
void h_report_on_device(int device_id) {

    // Report some information on a compute device
    cudaDeviceProp prop;

    // Get the properties of the compute device
    H_ERRCHK(cudaGetDeviceProperties(&prop, device_id));

    // ID of the compute device
    std::printf("Device id: %d\n", device_id);

    // Name of the compute device
    std::printf("\t%-40s %s\n","name:", prop.name);

    // Size of global memory
    std::printf("\t%-40s %lu MB\n","global memory size:", prop.totalGlobalMem/(1000000));

    // Maximum number of registers per block
    std::printf("\t%-40s %d \n","available registers per block:", prop.regsPerBlock);

    // Maximum shared memory size per block
    std::printf("\t%-40s %lu KB\n","maximum shared memory size per block:", prop.sharedMemPerBlock/(1000));

    // Maximum pitch size for memory copies (MB)
    std::printf("\t%-40s %lu MB\n","maximum pitch size for memory copies:", prop.memPitch/(1000000));

    // Print out the maximum number of threads along a dimension of a block
    std::printf("\t%-40s (", "max block size:");
    for (int n=0; n<2; n++) {
        std::printf("%d,", prop.maxThreadsDim[n]);
    }
    std::printf("%d)\n", prop.maxThreadsDim[2]); 
    std::printf("\t%-40s %d\n", "max threads in a block:", prop.maxThreadsPerBlock);
    
    // Print out the maximum size of a Grid
    std::printf("\t%-40s (", "max Grid size:");
    for (int n=0; n<2; n++) {
        std::printf("%d,", prop.maxGridSize[n]);
    }
    std::printf("%d)\n", prop.maxGridSize[2]); 
}

/// Release compute devices
void h_release_devices(int num_devices) {
    h_reset_devices(num_devices);
}

/// Create a number of streams
cudaStream_t* h_create_streams(size_t nstreams, int synchronise) {
    // Blocking is a boolean, 0==no, 
    assert(nstreams>0);

    unsigned int flag = cudaStreamDefault;

    // If blocking is 0 then set NonBlocking flag
    // meaning we don't synchronise with the null stream
    if (synchronise == 0) {
        flag = cudaStreamNonBlocking;
    }

    // Make the streams
    cudaStream_t* streams = (cudaStream_t*)calloc(nstreams, sizeof(cudaStream_t));

    for (int i=0; i<nstreams; i++) {
        H_ERRCHK(cudaStreamCreateWithFlags(&streams[i], flag));
    }

    return streams;
}

/// Release streams that were created
void h_release_streams(size_t nstreams, cudaStream_t* streams) {
    for (int i=0; i<nstreams; i++) {
        H_ERRCHK(cudaStreamDestroy(streams[i]));    
    }

    // Free streams array
    free(streams);
}

/// Get the IO rate in MB/s for bytes read or written
float h_get_io_rate_MBs(float elapsed_ms, size_t nbytes) {
    return (float)nbytes * 1.0e-3 / elapsed_ms;
}

/// Get how much time elapsed between two events that were recorded
float h_get_event_time_ms(
        // Assumes start and stop events have been recorded
        // with the cudaEventRecord() function
        cudaEvent_t t1,
        cudaEvent_t t2,
        const char* message, 
        size_t* nbytes) {
    
    // Make sure the stop and start events have finished
    H_ERRCHK(cudaEventSynchronize(t2));
    H_ERRCHK(cudaEventSynchronize(t1));

    // Elapsed time in milliseconds
    float elapsed_ms=0;

    // Convert the time into milliseconds
    H_ERRCHK(cudaEventElapsedTime(&elapsed_ms, t1, t2));
        
    // Print the timing message if necessary
    if ((message != NULL) && (strlen(message)>0)) {
        std::printf("Time for event \"%s\": %.3f ms", message, elapsed_ms);
        
        // Print transfer rate if nbytes is not NULL
        if (nbytes != NULL) {
            double io_rate_MBs = h_get_io_rate_MBs(
                elapsed_ms, 
                *nbytes
            );
            std::printf(" (%.2f MB/s)", io_rate_MBs);
        }
        std::printf("\n");
    }
    
    return elapsed_ms;
}

/// Make grid_nblocks big enough to fit a grid of at least global_size
void h_fit_blocks(dim3* grid_nblocks, dim3 global_size, dim3 block_size) {
    
    // Checks
    assert ((global_size.x>0) && (block_size.x>0));
    assert ((global_size.y>0) && (block_size.y>0));
    assert ((global_size.z>0) && (block_size.z>0));

    // Make the number of blocks
    (*grid_nblocks).x = global_size.x/block_size.x;
    if ((global_size.x % block_size.x)>0) {
        (*grid_nblocks).x += 1;
    }

    (*grid_nblocks).y = global_size.y/block_size.y;
    if ((global_size.y % block_size.y)>0) { 
        (*grid_nblocks).y += 1;
    }

    (*grid_nblocks).z = global_size.z/block_size.z;
    if ((global_size.z % block_size.z)>0) {
        (*grid_nblocks).z += 1;
    }
}

/// Allocate aligned memory for use on the host
void* h_alloc(size_t nbytes) {
    // Get the alignment to use by default
    size_t alignment = h_get_alignment();
    
#if defined(_WIN32) || defined(_WIN64)
    void* buffer = _aligned_malloc(nbytes, alignment);
#else
    void* buffer = aligned_alloc(alignment, nbytes);
#endif
    // Zero out the contents of the allocation for safety
    memset(buffer, '\0', nbytes);
    return buffer;
}

/// Open the file for reading and use std::fread to read in the file
void* h_read_binary(const char* filename, size_t *nbytes) {
    std::FILE *fp = std::fopen(filename, "rb");
    if (fp == NULL) {
        std::printf("Error in reading file %s", filename);
        exit(EXIT_FAILURE);
    }
    
    // Seek to the end of the file
    std::fseek(fp, 0, SEEK_END);
    
    // Extract the number of bytes in this file
    *nbytes = std::ftell(fp);

    // Rewind the file pointer
    std::rewind(fp);

    // Create a buffer to read into
    // Add an extra Byte for a null termination character
    // just in case we are reading to a string
    void *buffer = h_alloc((*nbytes)+1);
    
    // Set the NULL termination character for safety
    char* source = (char*)buffer;
    source[*nbytes] = '\0';
    
    // Read the file into the buffer and close
    size_t bytes_read = std::fread(buffer, 1, *nbytes, fp);
    assert(bytes_read == *nbytes);
    std::fclose(fp);
    return buffer;
}

/// Write binary data to file
void h_write_binary(void* data, const char* filename, size_t nbytes) {
    std::FILE *fp = std::fopen(filename, "wb");
    if (fp == NULL) {
        std::printf("Error in writing file %s", filename);
        exit(EXIT_FAILURE);
    }
    
    // Write the data to file
    std::fwrite(data, nbytes, 1, fp);
    
    // Close the file
    std::fclose(fp);
}

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
    cudaStream_t stream,
    // 0 for an ordered kernel launch, 1 for an out of order kernel launch
    int non_ordered_launch,
    // Function to prepare the kernel for launch
    // Needs flags for prep_kernel_function
    void (*prep_kernel_function)(const void*, void**, dim3, dim3, size_t*, void**),
    void** prep_kernel_args) {
    
        // Number of blocks in each dimension
    dim3 num_blocks = {1,1,1};
    
    // Fit the number of blocks
    h_fit_blocks(&num_blocks, global_size, block_size);
    
    // Prepare the kernel for execution, setting arguments etc
    if (prep_kernel_function!=NULL) {
        prep_kernel_function(kernel_function, kernel_args, num_blocks, block_size, shared_bytes, prep_kernel_args);
    }

    // cuda start and stop events
    cudaEvent_t t1=0, t2=0;
    // Create the events
    H_ERRCHK(cudaEventCreate(&t1));
    H_ERRCHK(cudaEventCreate(&t2));

    // Start event recording
    H_ERRCHK(cudaEventRecord(t1, stream));

    // Launch the kernel
    cudaError_t errcode = cudaLaunchKernel(
        kernel_function,
        num_blocks,
        block_size,
        kernel_args,
        *shared_bytes,
        stream
    );
    
    // Stop event recording
    H_ERRCHK(cudaEventRecord(t2, stream));

    // Elapsed milliseconds
    float elapsed = h_get_event_time_ms(t1, t2, NULL, NULL);
    
    // Destroy events
    H_ERRCHK(cudaEventDestroy(t1));
    H_ERRCHK(cudaEventDestroy(t2));
    
    if (errcode == cudaSuccess) {
        // Run was successful
        return elapsed;
    } else {
        // Manage the error
        printf(
            "Kernel execution not successful, error code is: %s\n", 
            cudaGetErrorString(errcode)
        );
        return FAULTY_TIME;
    }
}

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
        void** prep_kernel_args) {

    // Default block size
    dim3 temp_block_size = {16,1,1};
    if (default_block_size!=NULL) {
        temp_block_size.x = (*default_block_size).x;
        temp_block_size.y = (*default_block_size).y;
        temp_block_size.z = (*default_block_size).z;
    }

    // Maximum number of dimensions
    const int max_ndim=3;

    // Array of ints for block size (nexperiments, max_dims)
    // With row_major ordering
    uint32_t *input_block = NULL;
    size_t block_bytes = 0;

    // Look for the --block_file in argv, must be integer
    for (int i=1; i<argc; i++) {   
        if ((std::strncmp(argv[i], "--block_file", 12)==0) ||
            (std::strncmp(argv[i], "-block_file", 11)==0)) {
        
            // Read the input file as 32-bit integers
            input_block=(uint32_t*)h_read_binary("input_block.dat", &block_bytes);
        }
    }    
   
    // Get the device id
    int device_id=0;
    H_ERRCHK(cudaGetDevice(&device_id));

    // Report some information on a compute device
    cudaDeviceProp prop;

    // Get the properties of the current compute device
    H_ERRCHK(cudaGetDeviceProperties(&prop, device_id));

    // How many bytes do we use?
    size_t shared_bytes=0;

    if (input_block != NULL) {
        // Find the optimal block size 
        
        // Number of rows to process
        size_t nexperiments = block_bytes/(max_ndim*sizeof(uint32_t));
        
        // Input_block is of size (nexperiments, max_ndim)
        
        // Number of data points per experiment
        size_t npoints = 2; // (avg, stdev)
        // Output_block is of size (nexperiments, npoints)
        size_t nbytes_output = nexperiments*npoints*sizeof(double);
        double* output_block = (double*)malloc(nbytes_output);
        
        // Array to store the statistical timings for each experiment
        double* experiment_msec = new double[nstats];
        
        for (int n=0; n<nexperiments; n++) {
            // Run the application
            int nthreads = 1;
            int valid_size = 1;
            
            // Fill temp_block_size
            temp_block_size.x=(int)input_block[n*max_ndim+0];
            valid_size*=(temp_block_size.x<=prop.maxThreadsDim[0]);

            temp_block_size.y=(int)input_block[n*max_ndim+1];
            valid_size*=(temp_block_size.y<=prop.maxThreadsDim[1]);
            
            temp_block_size.z=(int)input_block[n*max_ndim+2];
            valid_size*=(temp_block_size.z<=prop.maxThreadsDim[2]);
            
            // Size of the block in threads
            nthreads = temp_block_size.x*temp_block_size.y*temp_block_size.z;

            // Average and standard deviation for statistical collection
            double avg=nan(""), stdev=nan("");

            if ((nthreads <= prop.maxThreadsPerBlock) && (valid_size > 0)) {
                // Run the experiment nstats times and get statistical information
                // Command queue must have profiling enabled 
                assert(nstats>3);
                
                std::printf("Running block size of: (%d, %d, %d)\n", temp_block_size.x, temp_block_size.y, temp_block_size.z);
                
                // Run function pointer here                
                for (int s=0; s<nstats; s++) {
                    experiment_msec[s] = (double)h_run_kernel(
                        kernel_function,
                        kernel_args,
                        global_size,
                        temp_block_size,
                        &shared_bytes,
                        // Use the null stream
                        0,
                        // Set the flag for an ordered launch
                        0,
                        // Function to prepare the data for use
                        prep_kernel_function,
                        prep_kernel_args
                     );
                }

                // Check the experiments for faulty kernel runs
                int errflag = 0;
                double t_lrg = 0.0;
                int index_lrg = 0;
                
                // Default avg and stdev
                double temp_avg=0.0;
                double temp_stdev=0.0;
                
                // Calculate the average, check for faulty kernel runs
                for (int s=0; s<nstats; s++) {
                    // Check for faulty kernel runs
                    if (experiment_msec[s]==FAULTY_TIME) errflag += 1;
                    // Check also for largest time
                    if (experiment_msec[s]>t_lrg) { 
                        t_lrg = experiment_msec[s];
                        index_lrg = s;
                    }
                    temp_avg+=experiment_msec[s];
                }
                
                // Remove the largest time
                temp_avg-=t_lrg;
                temp_avg/=(double)(nstats-1);
                // add prior times
                temp_avg+=prior_times;
                
                // Calculate standard deviation without contribution from faulty time
                for (int s=0; s<nstats; s++) {
                    if (s!=index_lrg) {
                        temp_stdev+=((experiment_msec[s]-temp_avg)*(experiment_msec[s]-temp_avg));
                    }
                }
                temp_stdev/=(double)(nstats-1);
                temp_stdev=sqrt(temp_stdev);
                
                if (errflag==0) {
                    // invalidate result
                    avg = temp_avg;
                    stdev = temp_stdev;
                }
            } 
                  
            // Send to output array 
            output_block[n*npoints+0] = avg;
            output_block[n*npoints+1] = stdev;
        }
                            
        h_write_binary(output_block, "output_block.dat", nbytes_output); 
                            
        delete[] experiment_msec;
        free(output_block);
        free(input_block);
        
    } else {
        // Run the kernel with just one experiment
        // Using the default block size
        double experiment_msec = h_run_kernel(
            kernel_function,
            kernel_args,
            global_size,
            temp_block_size,
            &shared_bytes,
            0,
            0,
            prep_kernel_function,
            prep_kernel_args
        );
        
        std::printf("Time for kernel was %.3f ms\n", experiment_msec);
    }
}

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
        cudaStream_t stream,
        // Starting offsets (in elements) for dst and src
        size_t offset_dst=0,
        size_t offset_src=0) {
    
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
    
    void** kernel_args = {0};
    kernel_args[0]=&dst;
    kernel_args[1]=&src;    
    kernel_args[2]=&offset_dst;
    kernel_args[3]=&offset_src;
    kernel_args[4]=&dims_dst;
    kernel_args[5]=&dims_src;
    kernel_args[6]=&region;
    
    // Run the kernel to copy things
    H_ERRCHK(cudaLaunchKernel(
        (const void*)h_copy_kernel3D,
        grid_nblocks,
        block_size,
        kernel_args,
        (size_t)0,
        stream
    ));
    
    
}