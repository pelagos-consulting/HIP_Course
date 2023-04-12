/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <iostream>
#include <atomic>

// Define the size of the arrays to be computed
#define NROWS_C 256
#define NCOLS_C 256

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
 
    // Read memory from the buffer to the host
    cl_uint value_T = 0;

    // Make some global memory for an atomic operation
    cl_mem buffer_T = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
        sizeof(cl_uint), 
        &value_T, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_T");

    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels_atomics.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, "");
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "atomics_test1", &errcode);
    h_errchk(errcode, "Creating Kernel");

    // Write memory from the host
    // to buffer_A and buffer_B on the compute device
    
    // Do we enable a blocking write?
    cl_bool blocking=CL_TRUE;
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Desired local size
    const size_t local_size[]={ 16, 16 };
    
    // Desired global_size
    const size_t global_size[]={ NROWS_C, NCOLS_C };

    h_errchk(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_T ),
        "setting kernel argument 0"
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
 
    h_errchk(
        clEnqueueReadBuffer(command_queue,
                            buffer_T,
                            blocking,
                            0,
                            sizeof(cl_uint),
                            &value_T,
                            1,
                            &kernel_event,
                            NULL), 
             "Copying T from device to host"
    );
    
    printf("A total of %u work items were executed\n", value_T);
    
    h_errchk(
        clReleaseMemObject(buffer_T),
        "releasing buffer T"
    );
    
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

