/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <iostream>

// Include the size of arrays to be computed
#include "mat_size.hpp"

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

typedef cl_float float_type;

void prep_mat_kernel(cl_kernel kernel, 
                 size_t* local_size,
                 size_t* global_size,
                 size_t ndim,
                 void* data) {
                 
    size_t* nbytes_line=(size_t*)data;
          
    // Set shared memory in argument 3
    // Local size is going to be (local_size[0], N1_A)
    h_errchk(
        clSetKernelArg(kernel, 3, local_size[0]*(*nbytes_line), NULL ),
        "setting kernel argument 3"
    );
    // Local size is going to be (local_size[1], N1_A)
    h_errchk(
        clSetKernelArg(kernel, 4, local_size[1]*(*nbytes_line), NULL ),
        "setting kernel argument 3"
    );                        
}

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

    // Do we enable blocking IO?
    cl_bool blocking = CL_TRUE;
    
    // Create the command queues
    cl_command_queue* command_queues = h_create_command_queues(
        devices,
        contexts,
        num_devices,
        num_command_queues,
        ordering,
        profiling
    );

    // Choose the first available context
    // and compute device to use
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
    float_type* array_A = (float_type*)h_read_binary("array_A.dat", &nbytes_A);
    float_type* array_B = (float_type*)h_read_binary("array_B.dat", &nbytes_B);

    // Sanity check on incoming data
    assert(nbytes_A==N0_C*N1_A*sizeof(float_type));   
    assert(nbytes_B==N1_A*N1_C*sizeof(float_type));
    nbytes_C=N0_C*N1_C*sizeof(float_type);
    
    // Make Buffers on the compute device for matrices A, B, and C
    float_type* array_C = (float_type*)h_alloc(nbytes_C);

    // Make buffer_A by copying from array_A
    cl_mem buffer_A = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
        nbytes_A, 
        (void*)array_A, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_A");
    
    // Make buffer_B using array_B as a backing store
    cl_mem buffer_B = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
        nbytes_B, 
        (void*)array_B, 
        &errcode
    );
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
    
    // Number of dimensions in the kernel
    size_t work_dim = 2;
    
    // Desired local size
    size_t local_size[]={ 4, 16 }; 
    
    // Desired global_size
    size_t global_size[]={ N0_C, N1_C };

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_mult_local", &errcode);
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
    
    // data for local memory preparation kernel
    size_t prep_data=N1_A*sizeof(float_type);
    
    // Prepare local memory arguments for execution
    prep_mat_kernel(
        kernel, 
        local_size,
        global_size,
        work_dim,
        &prep_data);
    
    h_errchk(
        clSetKernelArg(kernel, 5, sizeof(cl_uint), &N1_A ),
        "setting kernel argument 3"
    );
    h_errchk(
        clSetKernelArg(kernel, 6, sizeof(cl_uint), &N0_C ),
        "setting kernel argument 4"
    );
    h_errchk(
        clSetKernelArg(kernel, 7, sizeof(cl_uint), &N1_C ),
        "setting kernel argument 5"
    );
    

    
    // Number of statistical runs to do per experiment 
    size_t nstats = 3;
    

    
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
        0.0,
        prep_mat_kernel,
        &prep_data
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

