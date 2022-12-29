/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <chrono>
#include <iostream>

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

// Define the size of the arrays to be computed
#include "mat_size.hpp"

int main(int argc, char** argv) {
    // Start the clock
    auto time1 = std::chrono::high_resolution_clock::now();
    
    // Useful for checking OpenCL errors
    cl_int errcode;

    // Create handles to platforms, devices, and contexts
    cl_uint num_platforms;
    cl_uint num_devices;
    cl_platform_id *platforms = NULL;
    cl_device_id *devices = NULL;
    cl_context *contexts = NULL;

    // Discover platforms and devices and create contexts
    cl_device_type target_device=CL_DEVICE_TYPE_ALL;
    
    // Helper function to acquire devices
    h_acquire_devices(target_device,
                     &platforms,
                     &num_platforms,
                     &devices,
                     &num_devices,
                     &contexts);
    
    // Number of command queues to generate
    cl_uint num_command_queues = num_devices;
    
    // Create the command queues
    cl_command_queue* command_queues = h_create_command_queues(
        devices,
        contexts,
        num_devices,
        num_command_queues,
        CL_FALSE,
        CL_FALSE
    );

    // Choose the first available context and compute device to use
    cl_uint dev_index = 0;
    cl_context context = contexts[dev_index];
    cl_command_queue command_queue = command_queues[dev_index];
    cl_device_id device = devices[dev_index];
    
    // Report on the device in use
    h_report_on_device(device);
    
    // C is of size (N0_C, N1_C)
    
    cl_uint N0_C = NROWS_C, N1_C = NCOLS_C;
    size_t nbytes_C;

    nbytes_C=N0_C*N1_C*sizeof(cl_float);
    
    // Make an array to store the result in array_C
    cl_float* array_C = (cl_float*)calloc(nbytes_C, 1);
    
    // Make buffers for bringing data in and out of the computation
    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes_C, NULL, &errcode);
    h_errchk(errcode, "Creating buffer_C");

    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary("kernels.c", &nbytes_src);

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, "");
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "template", &errcode);
    h_errchk(errcode, "Creating Kernel");
    
    // Set arguments to the kernel (not thread safe)
    h_errchk(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_C ),"setting kernel argument 2");
    h_errchk(clSetKernelArg(kernel, 1, sizeof(cl_uint), &N0_C ),"setting kernel argument 4");
    h_errchk(clSetKernelArg(kernel, 2, sizeof(cl_uint), &N1_C ),"setting kernel argument 5");
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Desired local size
    const size_t local_work_size[]={ 16, 1 };
    
    // Desired global_size
    const size_t global_work_size[]={ N0_C, N1_C };
    
    // Enlarge the global size so that an integer number of local sizes fits within it
    h_fit_global_size(global_work_size, local_work_size, work_dim);
    
    // Event for the kernel
    cl_event kernel_event;
    
    // Now enqueue the kernel
    h_errchk(clEnqueueNDRangeKernel(command_queue,
                                    kernel,
                                    work_dim,
                                    NULL,
                                    global_work_size,
                                    local_work_size,
                                    0,
                                    NULL,
                                    &kernel_event), "Running the kernel");

    // Read memory from the buffer to the host
    h_errchk(clEnqueueReadBuffer(command_queue,
                            buffer_C,
                            CL_TRUE,
                            0,
                            nbytes_C,
                            array_C,
                            1,
                            &kernel_event,
                            NULL), "Copying matrix C from device to host");

    // Write out the result to file
    h_write_binary(array_C, "array_C.dat", nbytes_C);

    // Clean up memory that was allocated on the read   
    free(array_C);
    
    // Clean up command queues
    h_release_command_queues(command_queues, num_command_queues);
    
    // Clean up devices, queues, and contexts
    h_release_devices(
        devices,
        num_devices,
        contexts,
        platforms);

    // Stop the clock and get time elapsed
    auto time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<cl_double> elapsed_time = std::chrono::duration_cast<std::chrono::duration<cl_double>>(time2-time1);
    std::cout << "Elapsed time is " << elapsed_time.count() << "seconds" << std::endl;
}

