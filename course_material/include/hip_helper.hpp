///
/// @file  hip_helper.hpp
/// 
/// @brief Helper functions for HIP.
///
/// Written by Dr. Toby Potter 
/// for the Commonwealth Scientific and Industrial Research Organisation of Australia (CSIRO).
///

// Windows specific header instructions
#if defined(_WIN32) || defined(_WIN64)
    #define NOMINMAX
    #include <windows.h>
    #include <malloc.h>
#endif

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <cassert>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdint>

// Import the HIP header
#include <hip/hip_runtime.h>

/// Align all memory allocations to this byte boundary
#define BYTE_ALIGNMENT 64

/// Examine an error code and exit if necessary.
void h_errchk(hipError_t errcode, const char* message) {

    if (errcode != hipSuccess) { 
        const char* errstring = hipGetErrorString(errcode); 
        std::fprintf( 
            stderr, 
            "Error, HIP call failed at %s, error string is: %s\n", 
            message, 
            errstring 
        ); 
        exit(EXIT_FAILURE); 
    }
}

/// Macro to check error codes.
#define H_ERRCHK(cmd) \
{\
    h_errchk(cmd, "__FILE__:__LINE__");\
}

/// Get the least common multiple of two numbers.
size_t h_lcm(size_t n1, size_t n2) {
    size_t number = std::max(n1, n2);
    
    while ((number % n1) && (number % n2)) {
        number++;
    }
    
    return number;
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
    H_ERRCHK(hipGetDeviceCount(&max_devices));

    assert(dev_index<max_devices);
    assert(dev_index>=0);
    return(dev_index);
}

/// Initialise a primary context for all HIP devices
void h_acquire_devices(int* num_devices, int default_device_id) {
    
    // Initialise HIP 
    H_ERRCHK(hipInit(0));

    // Get the number of devices
    H_ERRCHK(hipGetDeviceCount(num_devices));

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
    H_ERRCHK(hipSetDevice(default_device_id));
}

/// Reset primary contexts on all compute devices
void h_reset_devices(int num_devices) {
    
    // Reset devices
    for (int i = 0; i<num_devices; i++) {
        // Set device
        H_ERRCHK(hipSetDevice(i));

        // Synchronize device 
        H_ERRCHK(hipDeviceSynchronize());

        // Reset device (destroys primary context)
        H_ERRCHK(hipDeviceReset());

        // Set the device to reinitialise it
        H_ERRCHK(hipSetDevice(i));
    }
}

/// Function to report information on a compute device
void h_report_on_device(int device_id) {

    // Report some information on a compute device
    hipDeviceProp_t prop;

    // Get the properties of the compute device
    H_ERRCHK(hipGetDeviceProperties(&prop, device_id));

    // ID of the compute device
    std::printf("Device id: %d\n", device_id);

    // Name of the compute device
    std::printf("\t%-40s %s\n","name:", prop.name);

    // Size of global memory
    std::printf("\t%-40s %lu MB\n","global memory size:",prop.totalGlobalMem/(1000000));

    // Maximum number of registers per block
    std::printf("\t%-40s %d \n","available registers per block:",prop.regsPerBlock);

    // Maximum shared memory size per block
    std::printf("\t%-40s %lu KB\n","maximum shared memory size per block:",prop.sharedMemPerBlock/(1000));

    // Maximum pitch size for memory copies (MB)
    std::printf("\t%-40s %lu MB\n","maximum pitch size for memory copies:",prop.memPitch/(1000000));

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
hipStream_t* h_create_streams(int nstreams, int blocking) {
    // Blocking is a boolean, 0==no, 
    assert(nstreams>0);

    unsigned int flag = hipStreamDefault;

    // If blocking is false then set NonBlocking flag
    if (blocking == 0) {
        flag = hipStreamNonBlocking;
    }

    // Make the streams
    hipStream_t* streams = (hipStream_t*)calloc((size_t)nstreams, sizeof(hipStream_t));

    for (int i=0; i<nstreams; i++) {
        H_ERRCHK(hipStreamCreateWithFlags(&streams[i], flag));
    }

    return streams;
}

/// Release streams that were created
void h_release_streams(int nstreams, hipStream_t* streams) {
    for (int i=0; i<nstreams; i++) {
        H_ERRCHK(hipStreamDestroy(streams[i]));    
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
        // with the hipEventRecord() function
        hipEvent_t t1,
        hipEvent_t t2,
        const char* message, 
        size_t* nbytes) {
    
    // Make sure the stop and start events have finished
    H_ERRCHK(hipEventSynchronize(t2));
    H_ERRCHK(hipEventSynchronize(t1));

    // Elapsed time in milliseconds
    float elapsed_ms=0;

    // Convert the time into milliseconds
    H_ERRCHK(hipEventElapsedTime(&elapsed_ms, t1, t2));
        
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
void* h_alloc(size_t nbytes, size_t alignment) {
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
    void *buffer = h_alloc((*nbytes)+1, BYTE_ALIGNMENT);
    
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
    // Number of blocks to run
    dim3 num_blocks,
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
    void** prep_kernel_args) {

    // Prepare the kernel for execution, setting arguments etc
    if (prep_kernel_function!=NULL) {
        prep_kernel_function(kernel_function, kernel_args, num_blocks, block_size, shared_bytes, prep_kernel_args);
    }

    // HIP start and stop events
    hipEvent_t t1=0, t2=0;
    // Create the events
    H_ERRCHK(hipEventCreate(&t1));
    H_ERRCHK(hipEventCreate(&t2));

    // Start event recording
    H_ERRCHK(hipEventRecord(t1, stream));

    // Launch the kernel
    H_ERRCHK(
        hipLaunchKernel(
            kernel_function,
            num_blocks,
            block_size,
            kernel_args,
            *shared_bytes,
            stream
        )
    );

    // Stop event recording
    H_ERRCHK(hipEventRecord(t2, stream));

    // Elapsed milliseconds
    float elapsed = h_get_event_time_ms(t1, t2, NULL, NULL);

    // Destroy events
    H_ERRCHK(hipEventDestroy(t1));
    H_ERRCHK(hipEventDestroy(t2));

    return elapsed;
}

/// Function to optimise the local size
/// if command line arguments are --local_file or -local_file.
/// Read an input file called input_local.dat of
/// type == uint32_t, and dimensions == (nexperiments, ndim) with row major ordering.
///
/// Writes to a file called output_local.dat of 
/// type == double and dimensions == (nexperiments, 2) with row major ordering
/// and where each line is (avg, stdev) in milliseconds
void h_optimise_local(
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

    // Default local size
    dim3 temp_block_size = {16,1,1};
    if (default_block_size!=NULL) {
        temp_block_size.x = (*default_block_size).x;
        temp_block_size.y = (*default_block_size).y;
        temp_block_size.z = (*default_block_size).z;
    }

    // Maximum number of dimensions
    const int max_ndim=3;

    // Array of ints for local size (nexperiments, max_dims)
    // With row_major ordering
    uint32_t *input_local = NULL;
    size_t local_bytes = 0;

    // Look for the --local_file in argv, must be integer
    for (int i=1; i<argc; i++) {   
        if ((std::strncmp(argv[i], "--local_file", 12)==0) ||
            (std::strncmp(argv[i], "-local_file", 11)==0)) {
        
            // Read the input file as 32-bit integers
            input_local=(uint32_t*)h_read_binary("input_local.dat", &local_bytes);
        }
    }    
   
    // Get the device id
    int device_id=0;
    H_ERRCHK(hipGetDevice(&device_id));

    // Report some information on a compute device
    hipDeviceProp_t prop;

    // Get the properties of the current compute device
    H_ERRCHK(hipGetDeviceProperties(&prop, device_id));

    // Number of blocks in each dimension
    dim3 temp_num_blocks = {1,1,1};

    // How many bytes do we use?
    size_t shared_bytes=0;

    if (input_local != NULL) {
        // Find the optimal local size 
        
        // Number of rows to process
        size_t nexperiments = local_bytes/(max_ndim*sizeof(uint32_t));
        
        // Input_local is of size (nexperiments, max_ndim)
        
        // Number of data points per experiment
        size_t npoints = 2; // (avg, stdev)
        // Output_local is of size (nexperiments, npoints)
        size_t nbytes_output = nexperiments*npoints*sizeof(double);
        double* output_local = (double*)malloc(nbytes_output);
        
        // Array to store the statistical timings for each experiment
        double* experiment_msec = new double[nstats];
        
        for (int n=0; n<nexperiments; n++) {
            // Run the application
            int nthreads = 1;
            int valid_size = 1;
            
            // Fill temp_block_size
            temp_block_size.x=(int)input_local[n*max_ndim+0];
            valid_size*=(temp_block_size.x<=prop.maxThreadsDim[0]);

            temp_block_size.y=(int)input_local[n*max_ndim+1];
            valid_size*=(temp_block_size.y<=prop.maxThreadsDim[1]);
            
            temp_block_size.z=(int)input_local[n*max_ndim+2];
            valid_size*=(temp_block_size.z<=prop.maxThreadsDim[2]);

            // Size of the block in threads
            nthreads = temp_block_size.x*temp_block_size.y*temp_block_size.z;
            
            // Fit the number of blocks
            h_fit_blocks(&temp_num_blocks, global_size, temp_block_size);

            // Average and standard deviation for statistical collection
            double avg=0.0, stdev=0.0;

            if ((nthreads <= prop.maxThreadsPerBlock) && (valid_size > 0)) {
                // Run the experiment nstats times and get statistical information
                // Command queue must have profiling enabled 

                // Run function pointer here                
                for (int s=0; s<nstats; s++) {
                    experiment_msec[s] = (double)h_run_kernel(
                        kernel_function,
                        kernel_args,
                        temp_num_blocks,
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

                // Calculate the average and standard deviation
                for (int s=0; s<nstats; s++) {
                    avg+=experiment_msec[s]+prior_times;
                }
                avg/=(double)nstats;
                
                for (int s=0; s<nstats; s++) {
                    stdev+=((experiment_msec[s]-avg)*(experiment_msec[s]-avg));
                }
                stdev/=(double)nstats;
                stdev=sqrt(stdev);
            } else {
                // No result
                avg = nan("");
                stdev = nan("");
            }
                  
            // Send to output array 
            output_local[n*npoints+0] = avg;
            output_local[n*npoints+1] = stdev;
        }
                            
        h_write_binary(output_local, "output_local.dat", nbytes_output); 
                            
        delete[] experiment_msec;
        free(output_local);
        free(input_local);
        
    } else {
        // Run the kernel with just one experiment
        // Using the default block size
        h_fit_blocks(&temp_num_blocks, global_size, temp_block_size);

        double experiment_msec = h_run_kernel(
            kernel_function,
            kernel_args,
            temp_num_blocks,
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
