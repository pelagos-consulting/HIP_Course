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

    // Do we enable blocking?
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
    // BT is of size (N1_C, N1_A)    
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
    
    // Make Buffers on the compute device for matrices A, B, BT, and C
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
    
    // Create buffer BT in the normal manner
    cl_mem buffer_BT = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE, 
        nbytes_B, 
        NULL, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_BT");
    
    // Allocate buffer C from pinned host memory
    cl_mem buffer_C = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE, 
        nbytes_C, 
        NULL, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_C");

    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels_mat_mult.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);
        
    // Write memory from the host
    // to buffer_A and buffer_B on the compute device
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Desired local size for all
    size_t local_size[]={ 16, 16 };
    
    // Create and run the transpose kernel
    cl_kernel kernel_transp=clCreateKernel(program, "transpose", &errcode);
    h_errchk(errcode, "Creating transpose kernel");
    
    // Desired global_size
    const size_t global_size_transp[]={ N1_A, N1_C };
    h_fit_global_size(global_size_transp, 
                      local_size, 
                      work_dim
    );    
    
    // Set kernel arguments
    h_errchk(
        clSetKernelArg(kernel_transp, 0, sizeof(cl_mem), &buffer_B ),
        "setting transpose kernel argument 0"
    );
    h_errchk(
        clSetKernelArg(kernel_transp, 1, sizeof(cl_mem), &buffer_BT ),
        "setting transpose kernel argument 1"
    );
    h_errchk(
        clSetKernelArg(kernel_transp, 2, sizeof(cl_uint), &N1_A ),
        "setting transpose kernel argument 2"
    );
    h_errchk(
        clSetKernelArg(kernel_transp, 3, sizeof(cl_uint), &N1_C ),
        "setting transpose kernel argument 3"
    );
    
    // Event for querying the transpose kernel
    cl_event kernel_event;

    // Now enqueue the transpose kernel
    h_errchk(
        clEnqueueNDRangeKernel(command_queue,
                                kernel_transp,
                                work_dim,
                                NULL,
                                global_size_transp,
                                local_size,
                                0,
                                NULL,
                                &kernel_event), 
        "Running the transpose kernel"
    );

    // Time the transpose kernel
    cl_double transpose_ms = h_get_event_time_ms(
        &kernel_event,
        NULL,
        NULL);        
    
    // Create and run the matrix multiplication kernel
    cl_kernel kernel_mat_mult=clCreateKernel(program, "mat_mult_transpose_B", &errcode);
    h_errchk(errcode, "Creating mat_mult_kernel");
        
    // Set arguments to the kernel (not thread safe)
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 0, sizeof(cl_mem), &buffer_A ),
        "setting kernel argument 0"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 1, sizeof(cl_mem), &buffer_BT ),
        "setting kernel argument 1"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 2, sizeof(cl_mem), &buffer_C ),
        "setting kernel argument 2"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 3, sizeof(cl_uint), &N1_A ),
        "setting kernel argument 5"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 4, sizeof(cl_uint), &N0_C ),
        "setting kernel argument 4"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 5, sizeof(cl_uint), &N1_C ),
        "setting kernel argument 5"
    );
    
    // Number of statistical runs to do per experiment 
    size_t nstats = 3;
    
    // Desired local size
    local_size[0] = 16;
    local_size[1] = 16;
    
    // Desired global_size
    size_t global_size[]={ N0_C, N1_C };
    
    // Run the optimisation program
    h_optimise_local(
        argc,
        argv,
        command_queue,
        kernel_mat_mult,
        device,
        global_size,
        local_size,
        work_dim,
        nstats,
        transpose_ms,
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
        clReleaseMemObject(buffer_BT),
        "releasing buffer BT"
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

