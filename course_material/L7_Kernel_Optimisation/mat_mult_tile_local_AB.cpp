/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

//// Step 1. Setup headers and parse command line arguments ////

#include <cassert>
#include <cmath>
#include <iostream>

// Bring in the size of the matrices
#include "mat_size.hpp"

// Bring in the library to work with matrices
#include "mat_helper.hpp"

// Bring in helper header to manage boilerplate code
#include "cl_helper.hpp"

typedef cl_float float_type;

const char* kernel_source = R"(

// Kernel function to get the start and end values
// for filling a shared memory array
void get_start_end(
    // Number of work-items along a dimension of workgroup
    size_t local_length,
    // Number of items in the array
    size_t array_length,
    // Index of work item along dimension of workgroup
    size_t local_index,
    // Starting position of the copy
    size_t *start,
    // End position of the copy
    size_t *end) {
  
    // Work out the jump size
    size_t jump_size=array_length/local_length;
    if (array_length%local_length) jump_size++;
    
    // Starting position for the copy
    *start=local_index*jump_size;
    // End position for the copy
    *end=(local_index+1)*jump_size;
    // Limit end so we don't go off the end
    *end=min(*end,array_length);
} 

// Matrix multiply kernel that uses local memory in a tiling way
__kernel void mat_mult_tile_local_AB (
                        __global float* A_star, 
                        __global float* B_star, 
                        __global float* C,
                        __local float* shared_A_star,
                        __local float* shared_B_star,
                        unsigned int N1_A_star, 
                        unsigned int N0_C,
                        unsigned int N1_C,
                        unsigned int chunk_len,
                        unsigned int start_chunk_id,
                        unsigned int end_chunk_id) { 
    
    // A_star is of size (N0_C, N1_A_star), (i0, n)
    // B_star is of size (N1_A_star, N1_C), (n, i1)
    // C is of size (N0_C, N1_C), (i0, i1)
    
    // i1 and i2 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i1=min(get_global_id(0), (size_t)N1_C-1); // Fastest dimension
    size_t i0=min(get_global_id(1), (size_t)N0_C-1); 
    
    // shared_A_star is of size (L0, chunk_len) (s0, n)
    // shared_B_star is of size (L1, chunk_len) (s1, n)
    size_t L0 = get_local_size(1); // Slowest dimension
    size_t L1 = get_local_size(0); // Fastest dimension
    
    // index within local memory
    size_t s0 = get_local_id(1); // Slowest dimension
    size_t s1 = get_local_id(0); // fastest dimension
    
    // Positions within shared memory
    __local float* shared_A_star_s0 = &shared_A_star[s0*chunk_len];
    __local float* shared_B_star_s1 = &shared_B_star[s1*chunk_len];

    // Scratch variable
    float temp=0.0f;

    // Start and end positions to copy within a chunk
    size_t start0, end0, start1, end1;
    get_start_end(L1, chunk_len, s1, &start1, &end1);
    get_start_end(L0, chunk_len, s0, &start0, &end0);

    // Loop over the chunks
    for (int chunk_id=start_chunk_id; chunk_id<end_chunk_id; chunk_id++) {

        // Fetch local memory into shared_A_star and shared_B_star
        
        // Starting positions for the copy
        __global float* A_star_i0 = &A_star[i0*N1_A_star+chunk_id*chunk_len];
        __global float* B_star_i1 = &B_star[chunk_id*chunk_len*N1_C+i1];
          
        // Fill the rows of shared_A_star and shared_B_star
        // Copy from row i0 of A_star
        for (size_t n = start1; n<end1; n++) {
            shared_A_star_s0[n] = A_star_i0[n];
        }
        
        // Copy from column i1 of B_star   
        for (size_t n = start0; n<end0; n++) {
            shared_B_star_s1[n] = B_star_i1[n*N1_C];
        }
              
        // Enqueue a local barrier to ensure shared memory is filled
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over columns of A and rows of B 
        for (size_t n=0; n<chunk_len; n++) {
                
            // Perform the dot product using local memory
            temp+=shared_A_star_s0[n]*shared_B_star_s1[n];
        }
        
        // Enqueue a local barrier to ensure all work items 
        // are ready to tackle the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Put the accumulated value into position
    C[i0*N1_C+i1]=temp;
}
)";

cl_int prep_mat_kernel(cl_kernel kernel, 
                 size_t* local_size,
                 size_t* global_size,
                 size_t ndim,
                 void* data) {
                 
    size_t* nbytes_line=(size_t*)data;
    
    cl_int errcode=CL_SUCCESS;

    // Set shared memory in argument 3
    // Local size of shared_A is going to be (local_size[1], chunk_len)
    errcode = errcode | clSetKernelArg(kernel, 3, local_size[1]*(*nbytes_line), NULL);

    // Local size of shared_B is going to be (local_size[0], chunk_len)
    errcode = errcode | clSetKernelArg(kernel, 4, local_size[0]*(*nbytes_line), NULL);                               
    return errcode;
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

    // Do we enable blocking IO?
    cl_bool blocking = CL_TRUE;
    
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
    // BT is of size (N1_C, N1_A)    
    // C is of size (N0_C, N1_C)
    
    //// Step 4. Prepare matrices A and B on the Host ////
    cl_uint N1_A = NCOLS_A, N0_C = NROWS_C, N1_C = NCOLS_C;

    // Number of bytes in each array
    size_t nbytes_A = N0_C*N1_A*sizeof(float_type);
    size_t nbytes_B = N1_A*N1_C*sizeof(float_type);
    size_t nbytes_C = N0_C*N1_C*sizeof(float_type);

    // Allocate memory for matrices A and B on the host
    float_type* A_h = (float_type*)h_alloc(nbytes_A);
    float_type* B_h = (float_type*)h_alloc(nbytes_B);

    // Fill A_h and B_h with random numbers 
    // using the matrix helper library
    m_random(A_h, N0_C, N1_A);
    m_random(B_h, N1_A, N1_C);

    // Get the cache line size
    cl_uint cache_line_bytes=64;

    H_ERRCHK(
        clGetDeviceInfo(
            device,
            CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
            sizeof(cl_uint),
            &cache_line_bytes,
            NULL
        )
    );
     
    // Sanity check the cache line size;
    cache_line_bytes = std::max(cache_line_bytes, (cl_uint)64);
    printf("Cache line size is %lu bytes\n", (long unsigned int)cache_line_bytes);
        
    //// Step 5. Allocate OpenCL Buffers for matrices A, B, and C ////
    
    // Number of elements we are going to use in a vector
    cl_uint chunk_len = 4*cache_line_bytes/sizeof(float_type);

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
    // C is of size (N0_C, N1_C)
    
    //// Step 5. Prepare OpenCL Buffers for matrices A, B, and C ////

    // Make A_star_d and B_star_d - the enlarged matrices
    cl_mem A_star_d = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE, 
        nbytes_A_star, 
        NULL, 
        &errcode
    );
    H_ERRCHK(errcode);
    
    cl_mem B_star_d = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE, 
        nbytes_B_star, 
        NULL, 
        &errcode
    );
    H_ERRCHK(errcode);
   
    // Allocate C from pinned host memory
    cl_mem C_d = clCreateBuffer(
        context, 
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 
        nbytes_C, 
        NULL, 
        &errcode
    );
    H_ERRCHK(errcode);

    // Zero out buffers A_star and B_star
    float_type zero=0.0;
    H_ERRCHK(
        clEnqueueFillBuffer(
            command_queue,
            A_star_d,
            &zero,
            sizeof(float_type),
            0,
            nbytes_A_star,
            0,
            NULL,
            NULL
        )
    );       
    H_ERRCHK(
        clEnqueueFillBuffer(
            command_queue,
            B_star_d,
            &zero,
            sizeof(float_type),
            0,
            nbytes_B_star,
            0,
            NULL,
            NULL
        )
    );
 
    // Rectangular copies from A_h and B_h 
    // to the enlarged arrays A_star_d and B_star_d
    
    // B is of size (N1_A, N1_C)
    // Offset is in bytes, row_id and slice_id are indices
    size_t offset=0, row_id=0, slice_id = 0;
    
    // Make up the origin for host and buffer, same for both
    const size_t dest_origin[] = {offset, row_id, slice_id};
    const size_t src_origin[] = {offset, row_id, slice_id};
    
    // Length of a row in the allocation (in bytes)
    size_t dest_row_pitch = N1_A_star * sizeof(float_type); 
    size_t src_row_pitch = N1_A * sizeof(float_type);
    
    // Number of bytes in a slice 
    size_t dest_slice_pitch = N0_C * dest_row_pitch;
    size_t src_slice_pitch = N0_C * src_row_pitch;        
        
    /// Size of the region to copy, of course we only copy 1 slice
    size_t nrows = N0_C, nslices = 1;
    const size_t region_A[] = {src_row_pitch, nrows, nslices};
     
    // Enqueue the rectangular copy of A_h to enlarged A_star_d
    H_ERRCHK(
        clEnqueueWriteBufferRect(
            command_queue,
            A_star_d,
            blocking,
            dest_origin,
            src_origin,
            region_A,
            dest_row_pitch,
            dest_slice_pitch,
            src_row_pitch,
            src_slice_pitch,
            A_h,
            0,
            NULL,
            NULL
        )
    );
   
    // Now copy B_h into enlarged B_star_d

    // Length of a row (in bytes)
    dest_row_pitch = N1_C * sizeof(float_type); 
    src_row_pitch = dest_row_pitch;
    
    // Number of bytes in a slice 
    dest_slice_pitch = N1_A_star * dest_row_pitch;
    src_slice_pitch = N1_A * src_row_pitch;        
        
    /// Size of the region to copy, only copy N1_A rows
    nrows = N1_A;
    const size_t region_B[] = {src_row_pitch, nrows, nslices};
     
    // Enqueue the rectangular copy
    H_ERRCHK(
        clEnqueueWriteBufferRect(
            command_queue,
            B_star_d,
            CL_TRUE,
            dest_origin,
            src_origin,
            region_B,
            dest_row_pitch,
            dest_slice_pitch,
            src_row_pitch,
            src_slice_pitch,
            B_h,
            0,
            NULL,
            NULL
        )
    );

    //// Step 6. Build the program from source for the chosen compute device ////

    // Turn this source code into a program
    cl_program program = h_build_program(kernel_source, context, device, NULL);
    
    //// Step 7. Create a kernel from the compiled program and set arguments ////
    
    // Event for querying kernels
    cl_event kernel_event;
    
    // Number of dimensions in the kernels
    size_t work_dim=2;

    // Desired local size
    size_t local_size[]={ 16, 16 };

    // Create the matrix multiplication kernel
    cl_kernel kernel_mat_mult=clCreateKernel(
        program, 
        "mat_mult_tile_local_AB", 
        &errcode
    );
    H_ERRCHK(errcode);
    
    size_t global_size_mat_mult[]={ N1_C, N0_C };

    // Set arguments to the kernel (not thread safe)
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 0, sizeof(cl_mem), &A_star_d ));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 1, sizeof(cl_mem), &B_star_d ));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 2, sizeof(cl_mem), &C_d ));

    // data for local memory preparation kernel
    size_t prep_data=chunk_len*sizeof(float_type);
    
    // Prepare local memory arguments for execution
    prep_mat_kernel(
        kernel_mat_mult, 
        local_size,
        global_size_mat_mult,
        work_dim,
        &prep_data
    );
    
    // Set kernel arguments
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 5, sizeof(cl_uint), &N1_A_star ));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 6, sizeof(cl_uint), &N0_C ));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 7, sizeof(cl_uint), &N1_C ));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 8, sizeof(cl_uint), &chunk_len ));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 9, sizeof(cl_uint), &start_chunk_id ));
    H_ERRCHK(clSetKernelArg(kernel_mat_mult, 10, sizeof(cl_uint), &end_chunk_id ));

    // Number of statistical runs per experiment
    size_t nstats=NSTATS;
    
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
        local_size,
        // Number of dimensions in the kernel
        work_dim,
        // Number of times to run the kernel per experiment
        nstats,
        // Any pre-existing timing results
        0.0,
        // Function for prepping the kernel prior to execution
        prep_mat_kernel,
        &prep_data
    );
    
    //// Step 10. Copy the Buffer for matrix C back to the host ////

    // Map C_d back to C_h so we can write it to disk
    float_type* C_h = (float_type*)clEnqueueMapBuffer(
        command_queue,
        C_d,
        blocking,
        CL_MAP_READ,
        0,
        nbytes_C,
        0,
        NULL,
        NULL,
        &errcode
    );
    H_ERRCHK(errcode);  

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
    h_write_binary(C_h, "array_C.dat", nbytes_C);

    //// Step 12. Clean up arrays and release resources

    // Unmap C_d so we can release it
    H_ERRCHK(
        clEnqueueUnmapMemObject(
            command_queue,
            C_d,
            C_h,
            0,
            NULL,
            NULL
        )
    );
    
    // Free the OpenCL buffers
    H_ERRCHK(clReleaseMemObject(A_star_d));
    H_ERRCHK(clReleaseMemObject(B_star_d));
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

