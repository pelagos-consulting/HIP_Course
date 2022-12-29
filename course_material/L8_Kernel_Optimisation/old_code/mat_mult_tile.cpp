/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

#include <cassert>
#include <cmath>
#include <sys/stat.h>
#include <iostream>

// Define the size of the arrays to be computed
#define NCOLS_A 1025
#define NROWS_C 1025
#define NCOLS_C 1025

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
    // BT is of size (N1_C, N1_A)    
    // C is of size (N0_C, N1_C)
    
    cl_uint N1_A = NCOLS_A, N0_C = NROWS_C, N1_C = NCOLS_C;
    size_t nbytes_A, nbytes_B, nbytes_C;

    // Read the input data into arrays and sanity check
    float_type* array_A = (float_type*)h_read_binary("array_A.dat", &nbytes_A);
    float_type* array_B = (float_type*)h_read_binary("array_B.dat", &nbytes_B);
    
    // Sanity check on incoming data
    assert(nbytes_A == N0_C*N1_A*sizeof(float_type));   
    assert(nbytes_B == N1_A*N1_C*sizeof(float_type));
    nbytes_C = N0_C*N1_C*sizeof(float_type);
    
    // Get the cache line size
    cl_uint cache_line_bytes=64;

    h_errchk(
        clGetDeviceInfo(
            device,
            CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
            sizeof(cl_uint),
            &cache_line_bytes,
            NULL
        ),
        "Getting the cache line size"
    );
     
    // Sanity check the cache line size;
    cache_line_bytes = std::max(cache_line_bytes, (cl_uint)64);
    printf("Cache line size is %lu\n", cache_line_bytes);
        
    // Number of elements we are going to use in a vector
    cl_uint chunk_len = cache_line_bytes/sizeof(float_type);

    // Integer (floored) number of vectors along axis of length N1_A
    cl_uint nchunks = N1_A/chunk_len;
    // Increase the number of vectors if there is any remainder
    if (N1_A % chunk_len) nchunks++;
    // Resized N1_A for allocation of B and A, may be larger than N1_A
    cl_uint N1_A_star = nchunks*chunk_len;

    // Set start and end chunk indices
    cl_uint start_chunk_id = 0;
    cl_uint end_chunk_id = nchunks;

    // Resized bytes due to enlarged N1_A_star
    size_t nbytes_A_star = N0_C*N1_A_star*sizeof(float_type);
    size_t nbytes_B_star = N1_C*N1_A_star*sizeof(float_type);    
    
    // A_star is of size (N0_C, N1_A_star)
    // B_star is of size (N1_A_star, N1_C)
    // BT_star is of size (N1_C, N1_A_star)
    // C is of size (N0_C, N1_C)

    
    // Make Buffers on the compute device for matrices A_star, B_star, BT_star, and C
    
    // Make buffer_A using array_A as a backing store
    cl_mem buffer_A_star = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
        nbytes_A_star, 
        NULL, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_A");
    
    // Make buffer_B using array_B as a backing store
    cl_mem buffer_B_star = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
        nbytes_B_star, 
        NULL, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_B");
    
    // Create buffer BT in the normal manner
    cl_mem buffer_BT_star = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE, 
        nbytes_B_star, 
        NULL, 
        &errcode
    );
    h_errchk(errcode, "Creating buffer_BT");
    
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
    
    // Zero out buffers A_star and B_star
    float_type zero=0.0;
    h_errchk(
        clEnqueueFillBuffer(
            command_queue,
            buffer_A_star,
            &zero,
            sizeof(float_type),
            0,
            nbytes_A_star,
            0,
            NULL,
            NULL
        ),
        "zero out buffer A_star"
    );       
    h_errchk(
        clEnqueueFillBuffer(
            command_queue,
            buffer_B_star,
            &zero,
            sizeof(float_type),
            0,
            nbytes_B_star,
            0,
            NULL,
            NULL
        ),
        "zero out buffer B_star"
    );
    
    // Rectangular copy to A and B
    
    // B is of size (N1_A, N1_C)
    // Offset is in bytes, row_id and slice_id are indices
    size_t offset=0, row_id=0, slice_id = 0;
    
    // Make up the origin for host and buffer, same for both
    const size_t buffer_origin[] = {offset, row_id, slice_id};
    const size_t host_origin[] = {offset, row_id, slice_id};
    
        // Length of a row in the allocation (in bytes)
    size_t buffer_row_pitch = N1_A_star * sizeof(float_type); 
    size_t host_row_pitch = N1_A * sizeof(float_type);
    
    // Number of bytes in a slice 
    size_t buffer_slice_pitch = N0_C * buffer_row_pitch;
    size_t host_slice_pitch = N0_C * host_row_pitch;        
        
    /// Size of the region to copy, of course we only copy 1 slice
    size_t nrows = N0_C, nslices = 1;
    const size_t region_A[] = {host_row_pitch, nrows, nslices};
     
    // Enqueue the rectangular copy
    h_errchk(
        clEnqueueWriteBufferRect(
            command_queue,
            buffer_A_star,
            CL_TRUE,
            buffer_origin,
            host_origin,
            region_A,
            buffer_row_pitch,
            buffer_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            array_A,
            0,
            NULL,
            NULL
        ),
        "Rectangular copy to buffer_A_star from the host"
    );
    
    // Length of a row (in bytes)
    buffer_row_pitch = N1_C * sizeof(float_type); 
    host_row_pitch = buffer_row_pitch;
    
    // Number of bytes in a slice 
    buffer_slice_pitch = N1_A_star * buffer_row_pitch;
    host_slice_pitch = N1_A * host_row_pitch;        
        
    /// Size of the region to copy, only copy N1_A rows
    nrows = N1_A;
    const size_t region_B[] = {host_row_pitch, nrows, nslices};
     
    // Enqueue the rectangular copy
    h_errchk(
        clEnqueueWriteBufferRect(
            command_queue,
            buffer_B_star,
            CL_TRUE,
            buffer_origin,
            host_origin,
            region_B,
            buffer_row_pitch,
            buffer_slice_pitch,
            host_row_pitch,
            host_slice_pitch,
            array_B,
            0,
            NULL,
            NULL
        ),
        "Rectangular copy to buffer_B_star from the host"
    );
    
    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, "");
        
    // Write memory from the host
    // to buffer_A and buffer_B on the compute device
    
    // Do we enable a blocking write?
    cl_bool blocking=CL_TRUE;
    
    // Number of dimensions in the kernel
    size_t work_dim_transp=2;
    
    // Desired local size for all
    const size_t local_size_transp[]={ 4, 32 };
    
    // Create the transpose kernel
    cl_kernel kernel_transp=clCreateKernel(
        program, 
        "transpose", 
        &errcode
    );
    h_errchk(errcode, "Creating transpose kernel");

    // Desired global_size
    const size_t global_size_transp[]={ N1_A_star, N1_C };
    h_fit_global_size(global_size_transp, 
                      local_size_transp, 
                      work_dim_transp
    );    
    
    // Set arguments for the transpose kernel
    h_errchk(
        clSetKernelArg(kernel_transp, 0, sizeof(cl_mem), &buffer_B_star ),
        "setting transpose kernel argument 0"
    );
    h_errchk(
        clSetKernelArg(kernel_transp, 1, sizeof(cl_mem), &buffer_BT_star ),
        "setting transpose kernel argument 1"
    );
    h_errchk(
        clSetKernelArg(kernel_transp, 2, sizeof(cl_uint), &N1_A_star ),
        "setting transpose kernel argument 2"
    );
    h_errchk(
        clSetKernelArg(kernel_transp, 3, sizeof(cl_uint), &N1_C ),
        "setting transpose kernel argument 3"
    );
    
    // Event for the transpose kernel
    cl_event event_transp;
    
    // Now enqueue the transpose kernel
    h_errchk(
        clEnqueueNDRangeKernel(command_queue,
                                kernel_transp,
                                work_dim_transp,
                                NULL,
                                global_size_transp,
                                local_size_transp,
                                0,
                                NULL,
                                &event_transp), 
        "Running the transpose kernel"
    );
    // Wait on the kernel to finish and time it takes
    cl_double run_transp_ms = h_get_event_time_ms(
        &event_transp, 
        "Running transpose kernel",
        NULL
    );
            
    // Create the matrix multiplication kernel
    cl_kernel kernel_mat_mult=clCreateKernel(
        program, 
        "mat_mult_tile", 
        &errcode
    );
    h_errchk(errcode, "Creating mat_mult_tile kernel.");
    
    // Desired global_size
    size_t work_dim_mat_mult = 2;
    
    size_t global_size_mat_mult[]={ N1_C, N0_C };
    size_t local_size_mat_mult[] = { 8, 8 };

    // Set arguments to the kernel (not thread safe)
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 0, sizeof(cl_mem), &buffer_A_star ),
        "setting kernel argument 0"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 1, sizeof(cl_mem), &buffer_BT_star ),
        "setting kernel argument 1"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 2, sizeof(cl_mem), &buffer_C ),
        "setting kernel argument 2"
    );    
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 3, sizeof(cl_uint), &N1_A_star ),
        "setting kernel argument 3"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 4, sizeof(cl_uint), &N0_C ),
        "setting kernel argument 4"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 5, sizeof(cl_uint), &N1_C ),
        "setting kernel argument 5"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 6, sizeof(cl_uint), &chunk_len ),
        "setting kernel argument 6"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 7, sizeof(cl_uint), &start_chunk_id ),
        "setting kernel argument 7"
    );
    h_errchk(
        clSetKernelArg(kernel_mat_mult, 8, sizeof(cl_uint), &end_chunk_id ),
        "setting kernel argument 8"
    );

    // Number of statistical runs per experiment
    size_t nstats=3;
    
    // Find the optimal local size
    h_optimise_local(
        argc,
        argv,
        command_queue,
        kernel_mat_mult,
        device,
        // Desired global size of the problem
        global_size_mat_mult,
        // Desired local_size of the problem, use NULL for defaults
        local_size_mat_mult,
        // Number of dimensions in the kernel
        work_dim_mat_mult,
        // Number of times to run the kernel per experiment
        nstats,
        run_transp_ms,
        // Function for prepping the kernel prior to execution
        NULL,
        NULL);
    
    // Map the buffer_C back to the host so we can write it to disk
    float_type* array_C = (float_type*)clEnqueueMapBuffer(
        command_queue,
        buffer_C,
        CL_TRUE,
        CL_MAP_READ,
        0,
        nbytes_C,
        0,
        NULL,
        NULL,
        &errcode
    );
    h_errchk(errcode, "Mapping matrix C from device to host");
    
    // Write out the result to file
    h_write_binary(array_C, "array_C.dat", nbytes_C);

    // Unmap buffer_C so we can release it
    h_errchk(
        clEnqueueUnmapMemObject(
            command_queue,
            buffer_C,
            array_C,
            0,
            NULL,
            NULL
        ),
        "Unmapping array C"
    );
    
    // Free the OpenCL buffers
    h_errchk(
        clReleaseMemObject(buffer_A_star),
        "releasing buffer A"
    );
    h_errchk(
        clReleaseMemObject(buffer_B_star),
        "releasing buffer B"
    );
    h_errchk(
        clReleaseMemObject(buffer_BT_star),
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

