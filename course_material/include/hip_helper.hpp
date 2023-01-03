
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

// Import the HIP header
#include <hip/hip_runtime.h>

// Function to check error code
#define h_errchk(errcode) { \
    hipError_t errcode; \
    if (errcode != hipSuccess) { \
        const char* errstring = hipGetErrorString(errcode); \
        std::fprintf( \
            stderr,  \ 
            "Error, HIP call failed at file %s, line %d\n Error string is: %s\n", \
            __FILE__, __LINE__, errstring \
        ); \
        exit(EXIT_FAILURE); \
    }\
}\


size_t h_lcm(size_t n1, size_t n2) {
    // Get the least common multiple of two numbers
    size_t number = std::max(n1, n2);
    
    while ((number % n1) && (number % n2)) {
        number++;
    }
    
    return number;
}

void h_show_options(const char* name) {
    // Display a helpful error message
    std::printf("Usage: %s <options> <DEVICE_INDEX>\n", name);
    std::printf("Options:\n");
    std::printf("\t-h,--help\t print help message\n");
    std::printf("\tDEVICE_INDEX is a number > 0\n"); 
}

int h_parse_args(int argc, char** argv) {
    
    // Parse command line arguments to extract device index to use
    
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
            dev_index = (cl_uint)std::atoi(arg);
        }
    }
    
    // Make sure the index is sane
    int max_devices=0;
    h_errchk(hipGetDeviceCount(&max_devices));

    assert(dev_index<max_devices);
    assert(dev_index>=0);
    return(dev_index);
}

// Function to clean devices before and after runtime
void h_clean_devices(int* num_devices_out) {
    
    // Get the number of devices
    int num_devices=0;
    h_errchk(hipGetDeviceCount(&num_devices));

    // Check to make sure we have one or more suitable devices
    if (num_devices == 0) {
        std::printf("Failed to find a suitable compute device\n");
        exit(EXIT_FAILURE);
    }

    // Reset devices
    for (int i = 0; i<num_devices; i++) {
        // Set device
        h_errchk(hipSetDevice(i));

        // Synchronize device 
        h_errchk(hipDeviceSynchronize());

        // Reset device (destroys primary context)
        h_errchk(hipDeviceReset());
    }

    // Set the number of output devices
    *num_devices_out = num_devices;
}

// Create streams
hipStream_t* h_create_streams(int nstreams, int blocking) {
    // Blocking is a boolean, 0==no, 
    assert(nstreams>0);

    unsigned int flag = hipStreamDefault;
    if (blocking == 0) {
        flag = hipStreamNonBlocking;
    }

    // Make the streams
    hipStream_t* streams = (hipStream_t*)calloc((size_t)nstreams, sizeof(hipStream_t));

    for (int i=0; i<nstreams; i++) {
        h_errchk(hipStreamCreateWithFlags(&streams[i], flag));
    }

    return streams;
}

void h_release_streams(int nstreams, hipStream_t* streams) {
    for (int i=0; i<nstreams; i++) {
        hipStreamDestroy(streams[i]);    
    }

    // Free streams array
    free(streams);
}


cl_double h_get_io_rate_MBs(cl_double time_ms, size_t nbytes) {
    // Get the IO rate in MB/s for bytes read or written
    return (cl_double)nbytes * 1.0e-3 / time_ms;
}

/// Got to here ///
cl_double h_get_event_time_ms(
        // Assumes start and stop events have been recorder

        cl_event *event, 
        const char* message, 
        size_t* nbytes) {
    
    // Make sure the event has finished
    h_errchk(clWaitForEvents(1, event), message);
    
    // Start and end times
    cl_ulong t1, t2;
        
    // Fetch the start and end times in nanoseconds
    h_errchk(
        clGetEventProfilingInfo(
            *event,
            CL_PROFILING_COMMAND_START,
            sizeof(cl_ulong),
            &t1,
            NULL
        ),
        "Fetching start time for event"
    );

    h_errchk(
        clGetEventProfilingInfo(
            *event,
            CL_PROFILING_COMMAND_END,
            sizeof(cl_ulong),
            &t2,
            NULL
        ),
        "Fetching end time for event"
    );
    
    // Convert the time into milliseconds
    cl_double elapsed = (cl_double)(t2-t1)*(cl_double)1.0e-6;
        
    // Print the timing message if necessary
    if ((message != NULL) && (strlen(message)>0)) {
        std::printf("Time for event \"%s\": %.3f ms", message, elapsed);
        
        // Print transfer rate if nbytes is not NULL
        if (nbytes != NULL) {
            cl_double io_rate_MBs = h_get_io_rate_MBs(
                elapsed, 
                *nbytes
            );
            std::printf(" (%.2f MB/s)", io_rate_MBs);
        }
        std::printf("\n");
    }
    
    return elapsed;
}

void h_fit_global_size(const size_t* global_size, const size_t* local_size, size_t work_dim) {
    // Fit global size so that an integer number of local sizes fits within it in any dimension
    
    // Make a readable pointer out of the constant one
    size_t* new_global = (size_t*)global_size;
    
    // Make sure global size is large enough
    for (int n=0; n<work_dim; n++) {
        assert(global_size[n]>0);
        assert(global_size[n]>=local_size[n]);
        if ((global_size[n] % local_size[n]) > 0) {
            new_global[n] = ((global_size[n]/local_size[n])+1)*local_size[n];
        } 
    }
}

void* h_alloc(size_t nbytes, size_t alignment) {
    // Allocate aligned memory for use on the host
#if defined(_WIN32) || defined(_WIN64)
    void* buffer = _aligned_malloc(nbytes, alignment);
#else
    void* buffer = aligned_alloc(alignment, nbytes);
#endif
    // Zero out the contents of the allocation for safety
    memset(buffer, '\0', nbytes);
    return buffer;
}

void* h_read_binary(const char* filename, size_t *nbytes) {
    // Open the file for reading and use std::fread to read in the file
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

void h_write_binary(void* data, const char* filename, size_t nbytes) {
    // Write binary data to file
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


// Function to report information on a compute device
void h_report_on_device(int device_id) {

    // Report some information on a compute device
    hipDeviceProp_t prop;

    // Get the properties of the compute device
    h_errchk(hipGetDeviceProperties(&prop, device_id));

    // Name
    std::printf("\t%20s %s \n","name:", prop.name);
    // Size of global memory
    std::printf("\t%20s %lu MB\n","global memory size:",prop.totalGlobalMem/(1000000));

    // Maximum number of registers per block
    std::printf("\t%20s %d \n","available registers per block:",prop.regsPerBlock);

    // Maximum shared memory size per block
    std::printf("\t%20s %lu KB\n","maximum shared memory size per block:",prop.sharedMemPerBlock/(1000));

    // Maximum pitch size for memory copies (MB)
    std::printf("\t%20s %lu MB\n","maximum pitch size for memory copies:",prop.memPitch/(1000000));

    // Print out the maximum number of threads along a dimension of a block
    std::printf("\t%20s (", "max block size:");
    for (int n=0; n<2; n++) {
        std::printf("%d,", prop.maxThreadsDim[n]);
    }
    std::printf("%d)\n", prop.maxThreadsDim[2]); 
    std::printf("\t%20s %d\n", "max threads in a block:", props.maxThreadsPerBlock);
    
    // Print out the maximum size of a Grid
    std::printf("\t%20s (", "max Grid size:");
    for (int n=0; n<2; n++) {
        std::printf("%d,", prop.maxGridSize[n]);
    }
    std::printf("%d)\n", prop.maxGridSize[2]); 
}

//
//// Function to run a kernel
//cl_double h_run_kernel(
//    cl_command_queue command_queue,
//    cl_kernel kernel,
//    size_t *local_size,
//    size_t *global_size,
//    size_t ndim,
//    cl_bool profiling,
//    // Function for prepping the kernel
//    void (*prep_kernel)(cl_kernel, size_t*, size_t*, size_t, void*),
//    void* prep_data
//    ) {
//    
//    // Sort out the global size
//    size_t *new_global = (size_t*)malloc(ndim*sizeof(size_t));
//    std::memcpy(new_global, global_size, ndim*sizeof(size_t));
//    h_fit_global_size(new_global, local_size, ndim);
//    
//    // Event management
//    cl_event kernel_event;
//    
//    // How much time did the kernel take?
//    cl_double elapsed_msec=0.0;
//    
//    // Prepare the kernel for execution, setting arguments etc
//    if (prep_kernel!=NULL) {
//        prep_kernel(kernel, local_size, new_global, ndim, prep_data);
//    }
//    
//    // Enqueue the kernel
//    cl_int errcode = clEnqueueNDRangeKernel(
//        command_queue,
//        kernel,
//        ndim,
//        NULL,
//        new_global,
//        local_size,
//        0,
//        NULL,
//        &kernel_event);
//    
//    // Profiling information
//    if ((profiling==CL_TRUE) && (errcode==CL_SUCCESS)) {
//        elapsed_msec = h_get_event_time_ms(
//            &kernel_event, 
//            NULL, 
//            NULL);
//    } else {
//        elapsed_msec = nan("");
//    }
//    
//    // Free allocated memory
//    free(new_global);
//    
//    return(elapsed_msec);
//}
//
//// Function to optimise the local size
//// if command line arguments are --local_file or -local_file
//// read an input file called input_local.dat
//// type == cl_uint and size == (nexperiments, ndim)
////
//// writes to a file called output_local.dat
//// type == cl_double and size == (nexperiments, 2)
//// where each line is (avd, stdev) in milli-seconds
//void h_optimise_local(
//        int argc,
//        char** argv,
//        cl_command_queue command_queue,
//        cl_kernel kernel,
//        cl_device_id device,
//        // Desired global size of the problem
//        size_t *global_size,
//        // Desired local_size of the problem, use NULL for defaults
//        size_t *local_size,
//        // Number of dimensions in the kernel
//        size_t ndim,
//        // Number of times to run the kernel per experiment
//        size_t nstats,
//        double prior_times,
//        void (*prep_kernel)(cl_kernel, size_t*, size_t*, size_t, void*),
//        void* prep_data) {
//    
//    // Maximum number of dimensions permitted
//    const int max_ndim = 3;
//    
//    // Terrible default for local_size
//    size_t temp_local_size[max_ndim] = {16,1,1};
//    if (local_size != NULL) {
//        for (size_t i=0; i<ndim; i++) {
//            temp_local_size[i] = local_size[i];
//        }
//    }
//    
//    // Array of ints for local size (nexperiments, max_dims)
//    // With row_major ordering
//    cl_uint *input_local = NULL;
//    size_t local_bytes = 0;
//
//    // Look for the --local_file in argv, must be integer
//    for (int i=1; i<argc; i++) {   
//        if ((std::strncmp(argv[i], "--local_file", 12)==0) ||
//            (std::strncmp(argv[i], "-local_file", 11)==0)) {
//        
//            // Read the input file
//            input_local=(cl_uint*)h_read_binary("input_local.dat", &local_bytes);
//        }
//    }    
//   
//    // Get the maximum number of work items in a work group
//    size_t max_work_group_size;
//    h_errchk(
//        clGetDeviceInfo(device, 
//                        CL_DEVICE_MAX_WORK_GROUP_SIZE, 
//                        sizeof(size_t), 
//                        &max_work_group_size, 
//                        NULL),
//        "Max number of work-items a workgroup."
//    );
//
//    // Get the maximum number of dimensions supported
//    cl_uint max_work_dims;
//    h_errchk(
//        clGetDeviceInfo(device, 
//                        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, 
//                        sizeof(cl_uint), 
//                        &max_work_dims, 
//                        NULL),
//        "Max number of dimensions for local size."
//    );
//    
//    // Make sure dimensions are good
//    assert(ndim<=max_work_dims);
//    
//    // Get the max number of work items along
//    // dimensions of a work group
//    size_t* max_size = new size_t[max_work_dims];
//    h_errchk(
//        clGetDeviceInfo(device, 
//                        CL_DEVICE_MAX_WORK_ITEM_SIZES, 
//                        max_work_dims*sizeof(size_t), 
//                        max_size, 
//                        NULL),
//        "Max size for work items."
//    );
//    
//    if (input_local != NULL) {
//        // Find the optimal local size 
//        
//        // Number of rows to process
//        size_t nexperiments = local_bytes/(max_ndim*sizeof(cl_uint));
//        
//        // Input_local is of size (nexperiments, max_ndim)
//        
//        // Number of data points per experiment
//        size_t npoints = 2; // (avg, stdev)
//        // Output_local is of size (nexperiments, npoints)
//        size_t nbytes_output = nexperiments*npoints*sizeof(cl_double);
//        cl_double* output_local = (cl_double*)malloc(nbytes_output);
//        
//        // Array to store the statistical timings for each experiment
//        cl_double* experiment_msec = new cl_double[nstats];
//        
//        for (int n=0; n<nexperiments; n++) {
//            // Run the application
//            size_t work_group_size = 1;
//            int valid_size = 1;
//            
//            // Fill local size
//            for (int i=0; i<max_ndim; i++) {
//                temp_local_size[i]=(size_t)input_local[n*max_ndim+i];
//                work_group_size*=temp_local_size[i];
//                // Check to make sure we aren't exceeding a limit
//                valid_size*=(temp_local_size[i]<=max_size[i]);
//            }
//            
//            // Average and standard deviation
//            cl_double avg=0.0, stdev=0.0;
//            
//            if ((work_group_size <= max_work_group_size) && (valid_size > 0)) {
//                // Run the experiment nstats times and get statistical information
//                // Command queue must have profiling enabled 
//                
//                // Run function pointer here
//                
//                for (int s=0; s<nstats; s++) {
//                    experiment_msec[s] = h_run_kernel(
//                        command_queue,
//                        kernel,
//                        temp_local_size,
//                        global_size,
//                        ndim,
//                        CL_TRUE,
//                        prep_kernel,
//                        prep_data
//                    );
//                }
//
//                // Calculate the average and standard deviation
//                for (int s=0; s<nstats; s++) {
//                    avg+=experiment_msec[s]+prior_times;
//                }
//                avg/=(cl_double)nstats;
//                
//                for (int s=0; s<nstats; s++) {
//                    stdev+=((experiment_msec[s]-avg)*(experiment_msec[s]-avg));
//                }
//                stdev/=(cl_double)nstats;
//                stdev=sqrt(stdev);
//            } else {
//                // No result
//                avg = nan("");
//                stdev = nan("");
//            }
//                  
//            // Send to output array 
//            output_local[n*npoints+0] = avg;
//            output_local[n*npoints+1] = stdev;
//        }
//                            
//        h_write_binary(output_local, "output_local.dat", nbytes_output); 
//                            
//        delete[] experiment_msec;
//        free(output_local);
//        free(input_local);
//    } else {
//        // Run the kernel with just one experiment
//        cl_double experiment_msec = h_run_kernel(
//                command_queue,
//                kernel,
//                temp_local_size,
//                global_size,
//                ndim,
//                CL_TRUE,
//                prep_kernel,
//                prep_data
//        );
//        
//        std::printf("Time for kernel was %.3f ms\n", experiment_msec);
//    }
//    
//    delete[] max_size;
//}
