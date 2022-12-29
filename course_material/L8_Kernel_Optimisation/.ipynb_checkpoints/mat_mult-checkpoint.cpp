/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <iostream>

// Define the size of the arrays to be computed
#define NCOLS_A 1025
#define NROWS_C 1025
#define NCOLS_C 1025

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

int main(int argc, char** argv) {
    
    // Parse arguments and set the target device
    cl_device_type target_device;
    cl_uint dev_index = h_parse_args(argc, argv, &target_device);
    
    // Useful for checking OpenCL errors
    cl_int errcode;

    // Create handles to platforms, 
    // devices, and contexts

    // Number of platforms discovered
    cl_uint num_platforms;

    // Number of devices discovered
    cl_uint num_devices;

    // Pointer to an array of platforms
    cl_platform_id *platforms = NULL;

    // Pointer to an array of devices
    cl_device_id *devices = NULL;

    // Pointer to an array of contexts
    cl_context *contexts = NULL;
    
    // Helper function to acquire devices
    h_acquire_devices(target_device,
                     &platforms,
                     &num_platforms,
                     &devices,
                     &num_devices,
                     &contexts);
    
    // Number of command queues to generate
    cl_uint num_command_queues = num_devices;
    
    // Do we enable out-of-order execution 
    cl_bool ordering = CL_FALSE;
    
    // Do we enable profiling?
    cl_bool profiling = CL_TRUE;
    
    // Create the command queues
    cl_command_queue* command_queues = h_create_command_queues(
        devices,
        contexts,
        num_devices,
        num_command_queues,
        ordering,
        profiling
    );

    // Make sure command line arguments are sane
    assert(dev_index < num_devices);
    cl_context context = contexts[dev_index];
    cl_command_queue command_queue = command_queues[dev_index];
    cl_device_id device = devices[dev_index];
    
    // Report on the device in use
    h_report_on_device(device);
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)    
    // C is of size (N0_C, N1_C)
    
    cl_uint N1_A = NCOLS_A, N0_C = NROWS_C, N1_C = NCOLS_C;
    size_t nbytes_A, nbytes_B, nbytes_C;

    // Read the input data into arrays and sanity check
    cl_float* array_A = (cl_float*)h_read_binary("array_A.dat", &nbytes_A);
    cl_float* array_B = (cl_float*)h_read_binary("array_B.dat", &nbytes_B);

    // Sanity check on incoming data
    assert(nbytes_A==N0_C*N1_A*sizeof(cl_float));   
    assert(nbytes_B==N1_A*N1_C*sizeof(cl_float));
    nbytes_C=N0_C*N1_C*sizeof(cl_float);
    
    // Make an array to store the result in array_C
    cl_float* array_C = (cl_float*)h_alloc(nbytes_C);
    
    // Make Buffers on the compute device for matrices A, B, and C
    cl_mem buffer_A = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_A, 
                                     NULL, 
                                     &errcode);
    h_errchk(errcode, "Creating buffer_A");
    
    cl_mem buffer_B = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_B, 
                                     NULL, 
                                     &errcode);
    h_errchk(errcode, "Creating buffer_B");
    
    cl_mem buffer_C = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_C, 
                                     NULL, 
                                     &errcode);
    h_errchk(errcode, "Creating buffer_C");

    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels_mat_mult.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_mult", &errcode);
    h_errchk(errcode, "Creating Kernel");
    
    // Set arguments to the kernel (not thread safe)
    h_errchk(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A ),
        "setting kernel argument 0"
    );
    h_errchk(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B ),
        "setting kernel argument 1"
    );
    h_errchk(
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C ),
        "setting kernel argument 2"
    );
    h_errchk(
        clSetKernelArg(kernel, 3, sizeof(cl_uint), &N1_A ),
        "setting kernel argument 3"
    );
    h_errchk(
        clSetKernelArg(kernel, 4, sizeof(cl_uint), &N0_C ),
        "setting kernel argument 4"
    );
    h_errchk(
        clSetKernelArg(kernel, 5, sizeof(cl_uint), &N1_C ),
        "setting kernel argument 5"
    );

    // Write memory from the host
    // to buffer_A and buffer_B on the compute device
    
    // Do we enable a blocking write?
    cl_bool blocking=CL_TRUE;
    
    h_errchk(
        clEnqueueWriteBuffer(command_queue,
                            buffer_A,
                            blocking,
                            0,
                            nbytes_A,
                            array_A,
                            0,
                            NULL,
                            NULL), 
        "Writing to buffer_A from host"
    );

    h_errchk(
        clEnqueueWriteBuffer(command_queue,
                            buffer_B,
                            blocking,
                            0,
                            nbytes_B,
                            array_B,
                            0,
                            NULL,
                            NULL), 
        "Writing to buffer_B from host"
    );
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Number of statistical runs to do per experiment 
    size_t nstats = 3;
    
    // Desired local size
    size_t local_size[]={ 8, 8 };
    
    // Desired global_size
    size_t global_size[]={ N0_C, N1_C };
    
    // Run the optimisation program
    h_optimise_local(
        argc,
        argv,
        command_queue,
        kernel,
        device,
        global_size,
        local_size,
        work_dim,
        nstats,
        NULL,
        NULL
    );
    
    // Read memory from the buffer to the host
    h_errchk(
        clEnqueueReadBuffer(command_queue,
                            buffer_C,
                            blocking,
                            0,
                            nbytes_C,
                            array_C,
                            0,
                            NULL,
                            NULL), 
             "Copying matrix C from device to host"
    );
    
    // Write out the result to file
    h_write_binary(array_C, "array_C.dat", nbytes_C);

    // Free the OpenCL buffers
    h_errchk(
        clReleaseMemObject(buffer_A),
        "releasing buffer A"
    );
    h_errchk(
        clReleaseMemObject(buffer_B),
        "releasing buffer B"
    );
    h_errchk(
        clReleaseMemObject(buffer_C),
        "releasing buffer C"
    );
    
    // Clean up memory that was allocated on the read   
    free(array_A);
    free(array_B);
    free(array_C);
    
    // Clean up command queues
    h_release_command_queues(
        command_queues, 
        num_command_queues
    );
    
    // Clean up devices, queues, and contexts
    h_release_devices(
        devices,
        num_devices,
        contexts,
        platforms
    );
    
    return 0;
}

