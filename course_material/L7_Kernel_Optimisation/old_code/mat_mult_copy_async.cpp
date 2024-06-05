/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <iostream>
#include <chrono>

// Define the size of the arrays to be computed
#define NCOLS_A 256
#define NROWS_C 520
#define NCOLS_C 1032

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
    cl_bool profiling = CL_FALSE;
    
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
    cl_float* array_A = (cl_float*)h_read_binary("array_A.dat", &nbytes_A);
    cl_float* array_B = (cl_float*)h_read_binary("array_B.dat", &nbytes_B);

    // Sanity check on incoming data
    assert(nbytes_A==N0_C*N1_A*sizeof(cl_float));   
    assert(nbytes_B==N1_A*N1_C*sizeof(cl_float));
    nbytes_C=N0_C*N1_C*sizeof(cl_float);
        
    // Make Buffers on the compute device for matrices A, B, and C
    cl_float* array_C = (cl_float*)h_alloc(nbytes_C);
    
    // Make buffer_A using array_A as a backing store
    cl_mem buffer_A = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
        nbytes_A, 
        (void*)array_A, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_A");
    
    // Create buffer B in the normal manner
    cl_mem buffer_B = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE, 
        nbytes_B, 
        NULL, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_B");
    
    // Allocate buffer C from pinned host memory
    cl_mem buffer_C = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
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
    cl_program program = h_build_program(kernel_source, context, device, "");
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_mult_local", &errcode);
    h_errchk(errcode, "Creating Kernel");

    // Write memory from the host
    // to buffer_A and buffer_B on the compute device
    
    // Do we enable a blocking write?
    cl_bool blocking=CL_TRUE;
    
    // We don't need to copy buffer_A across because we use the host_pointer

    // Do a rectangular copy from host to memory in buffer_B
    
    // B is of size (N1_A, N1_C)
    // Offset is in bytes, row_id and slice_id are indices
    size_t offset=0, row_id=0, slice_id = 0;
    
    // Make up the origin for host and buffer
    const size_t buffer_origin[] = {offset, row_id, slice_id};
    const size_t host_origin[] = {offset, row_id, slice_id};
    
    // Length of a row (in bytes)
    size_t buffer_row_pitch = N1_C * sizeof(cl_float); 
    size_t host_row_pitch = buffer_row_pitch;
    
    // Number of bytes in a slice 
    size_t buffer_slice_pitch = N1_A * buffer_row_pitch;
    size_t host_slice_pitch = N1_A * host_row_pitch;        
        
    /// Size of the region to copy, of course we only copy 1 slice
    size_t nrows = N1_A, nslices = 1;
    const size_t region[] = {buffer_row_pitch, nrows, nslices};
     
    // Enqueue the rectangular copy
    h_errchk(
        clEnqueueWriteBufferRect(
            command_queue,
            buffer_B,
            CL_TRUE,
            buffer_origin,
            host_origin,
            region,
            buffer_row_pitch,
            buffer_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            array_B,
            0,
            NULL,
            NULL
        ),
        "Rectangular copy to buffer_B from the host"
    );
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Desired local size
    const size_t local_size[]={ 4, 16 };
    
    // Desired global_size
    const size_t global_size[]={ N0_C, N1_C };
    
    // Enlarge the global size so that 
    // an integer number of local sizes fits within it
    h_fit_global_size(global_size, 
                      local_size, 
                      work_dim
    );
    
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
    // Set shared memory in argument 3
    // Local size is going to be (local_size[0], N1_A)
    h_errchk(
        clSetKernelArg(kernel, 3, local_size[0]*N1_A*sizeof(cl_float), NULL ),
        "setting kernel argument 3"
    );
    // Local size is going to be (local_size[1], N1_A)
    h_errchk(
        clSetKernelArg(kernel, 4, local_size[1]*N1_A*sizeof(cl_float), NULL ),
        "setting kernel argument 3"
    );
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
    
    // Event for the kernel
    cl_event kernel_event;
    
    // Now enqueue the kernel
    h_errchk(
        clEnqueueNDRangeKernel(command_queue,
                                kernel,
                                work_dim,
                                NULL,
                                global_size,
                                local_size,
                                0,
                                NULL,
                                &kernel_event), 
        "Running the kernel"
    );

    // Wait on the kernel to finish
    h_errchk(
        clWaitForEvents(1, &kernel_event),
        "Waiting on the kernel"
    );
    
    // Time how long it takes to do the buffer read
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Read memory from the buffer to the host
    h_errchk(
        clEnqueueReadBuffer(command_queue,
                            buffer_C,
                            CL_FALSE,
                            0,
                            nbytes_C,
                            array_C,
                            1,
                            &kernel_event,
                            NULL), 
             "Copying matrix C from device to host"
    );
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> interval = t2 - t1;
    printf("Time taken to copy is %f ms\n", interval*1000.0);
    
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
}

