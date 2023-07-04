/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
//#include <sys/stat.h>
#include <iostream>

// Bring in the size of the matrices
#include "mat_size.hpp"

// Bring in the library to work with matrices
#include "mat_helper.hpp"

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
    
    //// Step 2. Discover resources ////
    
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
    
    //// Step 3. Allocate command queues and choose a compute device ////
    
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
    // Also make sure command line arguments are sane
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
    
    //// Step 4. Prepare matrices A, B, and C on the Host ////
    cl_uint N1_A = NCOLS_A, N0_C = NROWS_C, N1_C = NCOLS_C;
    
    // Assert that N1_A must be a multiple of 8 in order to use float8 in the kernel
    // Otherwise we have to resize A and B so that N1_A%8 == 0
    size_t vector_len=8;
    assert(N1_A%vector_len==0);
    // Reduced vector length
    cl_uint N1_A_v = N1_A/vector_len;

    // Number of bytes in each array
    size_t nbytes_A = N0_C*N1_A*sizeof(cl_float);
    size_t nbytes_B = N1_A*N1_C*sizeof(cl_float);
    size_t nbytes_C = N0_C*N1_C*sizeof(cl_float);

    // Allocate memory for matrices A, B, and C on the host
    cl_float* A_h = (cl_float*)h_alloc(nbytes_A);
    cl_float* B_h = (cl_float*)h_alloc(nbytes_B);

    // Fill A_h and B_h with random numbers 
    // using the matrix helper library
    m_random(A_h, N0_C, N1_A);
    m_random(B_h, N1_A, N1_C);
        
    //// Step 5. Allocate OpenCL Buffers for matrices A, B, and C ////
    
    // Make A_d by copying from A_h
    cl_mem A_d = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
        nbytes_A, 
        (void*)A_h, 
        &errcode
    );
    H_ERRCHK(errcode);
    
    // Create buffer B_d in the normal manner
    cl_mem B_d = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE, 
        nbytes_B, 
        NULL, 
        &errcode
    );
    H_ERRCHK(errcode);
    
    // Fill buffer B_d with ones as an example of clEnqueueFillBuffer
    cl_float one = 1.0;
    H_ERRCHK(
        clEnqueueFillBuffer(
            command_queue,
            B_d,
            &one,
            sizeof(cl_float),
            0,
            nbytes_B,
            0,
            NULL,
            NULL
        )
    );
    
    // Allocate buffer C from pinned host memory
    cl_mem C_d = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
        nbytes_C, 
        NULL, 
        &errcode
    );
    H_ERRCHK(errcode);

    //// Step 6. Build the program from source for the chosen compute device ////
    
    // Now specify the kernel source and read it in
    size_t nbytes_src = 0;
    const char* kernel_source = (const char*)h_read_binary(
        "kernels_mat_mult.c", 
        &nbytes_src
    );

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);
        
    // Create a kernel from the built program
    cl_kernel kernel=clCreateKernel(program, "mat_mult_local_vector", &errcode);
    H_ERRCHK(errcode);

    //// Step 8. Upload matrices A and B from the host to the OpenCL device Buffers ////
    
    // Write memory from the host
    // to A_d and B_d on the compute device
    
    // Do we enable a blocking write?
    cl_bool blocking=CL_TRUE;
    
    // We don't need to copy A_d across because we use the host_pointer

    // Do a rectangular copy from host to memory in B_d
    
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
    H_ERRCHK(
        clEnqueueWriteBufferRect(
            command_queue,
            B_d,
            blocking,
            buffer_origin,
            host_origin,
            region,
            buffer_row_pitch,
            buffer_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            B_h,
            0,
            NULL,
            NULL
        )
    );
    
    //// Step 9. Run the kernel to compute C from A and B ////
    
    // Number of dimensions in the kernel
    size_t work_dim=2;
    
    // Desired local size
    const size_t local_size[]={ 8, 8 };
    
    // Desired global_size
    const size_t global_size[]={ N1_C, N0_C };
    
    // Enlarge the global size so that 
    // an integer number of local sizes fits within it
    h_fit_global_size(global_size, 
                      local_size, 
                      work_dim
    );
    
    // Set arguments to the kernel (not thread safe)
    H_ERRCHK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &A_d ));
    H_ERRCHK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &B_d ));
    H_ERRCHK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &C_d ));
    // Set shared memory in argument 3 for shared_B
    // Local size is going to be (local_size[0], N1_A)
    H_ERRCHK(clSetKernelArg(kernel, 3, local_size[0]*N1_A*sizeof(cl_float), NULL ));
    H_ERRCHK(clSetKernelArg(kernel, 4, sizeof(cl_uint), &N1_A ));
    H_ERRCHK(clSetKernelArg(kernel, 5, sizeof(cl_uint), &N0_C ));
    H_ERRCHK(clSetKernelArg(kernel, 6, sizeof(cl_uint), &N1_C ));
    H_ERRCHK(clSetKernelArg(kernel, 7, sizeof(cl_uint), &N1_A_v ));
    
    // Event for the kernel
    cl_event kernel_event;
    
    // Now enqueue the kernel
    H_ERRCHK(
        clEnqueueNDRangeKernel(
            command_queue,
            kernel,
            work_dim,
            NULL,
            global_size,
            local_size,
            0,
            NULL,
            &kernel_event
        )
    );

    // Wait on the kernel to finish
    H_ERRCHK(clWaitForEvents(1, &kernel_event));
    
    // Map C_d back to the host so we can write it to disk
    cl_float* C_h = (cl_float*)clEnqueueMapBuffer(
        command_queue,
        C_d,
        CL_TRUE,
        CL_MAP_READ,
        0,
        nbytes_C,
        0,
        NULL,
        NULL,
        &errcode
    );
    H_ERRCHK(errcode);
    
    // Write C_h to disk
    h_write_binary(C_h, "array_C.dat", nbytes_C);
    
    // Unmap C_d so we can release it
    H_ERRCHK(
        clEnqueueUnmapMemObject(
            command_queue,
            C_d,
            (void*)C_h,
            0,
            NULL,
            NULL
        )
    );
    
    //// Step 11. Test the answer against a known solution
    //// And write the contents of the matrices out to disk
   
    // Compute the serial solution using the matrix helper library
    float* C_answer_h = (float*)calloc(nbytes_C, 1);
    m_mat_mult(A_h, B_h, C_answer_h, N1_A, N0_C, N1_C);

    // Print the maximum error between matrices
    float max_err = m_max_error(C_h, C_answer_h, N0_C, N1_C);

    // Write out the host arrays to file
    h_write_binary(A_h, "array_A.dat", nbytes_A);
    h_write_binary(B_h, "array_B.dat", nbytes_B);

    //// Step 12. Clean up arrays and release resources
    
    // Free the OpenCL buffers
    H_ERRCHK(clReleaseMemObject(A_d));
    H_ERRCHK(clReleaseMemObject(B_d));
    H_ERRCHK(clReleaseMemObject(C_d));
    
    // Clean up memory that was allocated on the read   
    free(A_h);
    free(B_h);
    free(C_answer_h);
    
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

