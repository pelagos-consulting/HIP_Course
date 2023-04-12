/* Code to perform Hadamard (elementwise) multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <iostream>

// Define the size of the arrays to be computed
#define NROWS_F 520
#define NCOLS_F 1032

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
    
    // Check if the device supports fine-grained buffer SVM
    cl_device_svm_capabilities svm;
    errcode = clGetDeviceInfo(
        device,
        CL_DEVICE_SVM_CAPABILITIES,
        sizeof(cl_device_svm_capabilities),
        &svm,
        NULL
    );
    
    if (errcode == CL_SUCCESS && 
        (svm & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
        printf("Device supports fine-grained buffer SVM\n");
    } else {
        printf("Sorry, this device can not support fine-grained buffer SVM\n");
        printf("No solution performed\n");
        exit(OCL_EXIT);
    }
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    
    // D, E, F is of size (N0_F, N1_F)
    cl_uint N0_F = NROWS_F, N1_F = NCOLS_F;
    size_t nbytes_D, nbytes_E, nbytes_F;

    // Read the input data into arrays and sanity check
    cl_float* array_D = (cl_float*)h_read_binary("array_D.dat", &nbytes_D);
    cl_float* array_E = (cl_float*)h_read_binary("array_E.dat", &nbytes_E);

    // Sanity check on incoming data
    assert(nbytes_D==N0_F*N1_F*sizeof(cl_float));   
    assert(nbytes_E==N0_F*N1_F*sizeof(cl_float));
    nbytes_F=N0_F*N1_F*sizeof(cl_float);
    
    // Allocate SVM memory for array F
    cl_float *array_F = (cl_float*)clSVMAlloc(
        context,
        CL_MEM_WRITE_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER,
        nbytes_F,
        0
    );
    
    // Make Buffers on the compute device for matrices D, E, and F
    cl_mem buffer_D = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_D, 
                                     NULL, 
                                     &errcode);
    h_errchk(errcode, "Creating buffer_D");
    
    cl_mem buffer_E = clCreateBuffer(context, 
                                     CL_MEM_READ_WRITE, 
                                     nbytes_E, 
                                     NULL, 
                                     &errcode);
    h_errchk(errcode, "Creating buffer_E");
    
    // No need to create a buffer for F

    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels_elementwise.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_elementwise", &errcode);
    h_errchk(errcode, "Creating Kernel");
    
    // Set arguments to the kernel (not thread safe)
    h_errchk(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_D ),
        "setting kernel argument 0"
    );
    h_errchk(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_E ),
        "setting kernel argument 1"
    );
    // Need to replace clSetKernelArg 
    // with clSetKernelArgSVMPointer for argument 2
    h_errchk(
        clSetKernelArgSVMPointer(
            kernel, 2, array_F
        ),
        "setting kernel argument 2"
    );
    h_errchk(
        clSetKernelArg(kernel, 3, sizeof(cl_uint), &N0_F ),
        "setting kernel argument 3"
    );
    h_errchk(
        clSetKernelArg(kernel, 4, sizeof(cl_uint), &N1_F ),
        "setting kernel argument 4"
    );

    // Write memory from the host
    // to buffer_D and buffer_E on the compute device
    
    // Do we enable a blocking write?
    cl_bool blocking=CL_TRUE;
    
    h_errchk(
        clEnqueueWriteBuffer(command_queue,
                            buffer_D,
                            blocking,
                            0,
                            nbytes_D,
                            array_D,
                            0,
                            NULL,
                            NULL), 
        "Writing to buffer_D from host"
    );

    h_errchk(
        clEnqueueWriteBuffer(command_queue,
                            buffer_E,
                            blocking,
                            0,
                            nbytes_E,
                            array_E,
                            0,
                            NULL,
                            NULL), 
        "Writing to buffer_E from host"
    );
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Desired local size
    const size_t local_size[]={ 16, 1 };
    
    // Desired global_size
    const size_t global_size[]={ N0_F, N1_F };
    
    // Enlarge the global size so that 
    // an integer number of local sizes fits within it
    h_fit_global_size(global_size, 
                      local_size, 
                      work_dim
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
    
    // At this point F is available
    // for access by the host
    // No need to copy F back to the host
    
    // Write out the result to file
    h_write_binary(array_F, "array_F.dat", nbytes_F);

    // Free the OpenCL buffers
    h_errchk(
        clReleaseMemObject(buffer_D),
        "releasing buffer D"
    );
    h_errchk(
        clReleaseMemObject(buffer_E),
        "releasing buffer E"
    );

    // No need to release buffer_F
    
    // Clean up memory that was allocated on the read   
    free(array_D);
    free(array_E);
    
    // Need to clSVMFree array_F
    clSVMFree(context, array_F);
    
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

