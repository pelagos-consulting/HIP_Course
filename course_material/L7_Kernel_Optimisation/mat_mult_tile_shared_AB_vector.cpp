/* Code to perform a Matrix multiplication using HIP
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
#include "hip_helper.hpp"

typedef float float_type;
typedef float4 float_vector_type;

// Device function to get the start and end values
// for filling a shared memory array
__device__ void get_start_end(
    // Number of work-items along a dimension of workgroup
    size_t block_length,
    // Number of items in the array
    size_t array_length,
    // Index of work item along dimension of workgroup
    size_t block_index,
    // Starting position of the copy
    size_t *start,
    // End position of the copy
    size_t *end) {
  
    // Work out the jump size
    size_t jump_size=array_length/block_length;
    if (array_length%block_length) jump_size++;
    
    // Starting position for the copy
    *start=block_index*jump_size;
    // End position for the copy
    *end=(block_index+1)*jump_size;
    // Limit end so we don't go off the end
    *end=min(*end,array_length);
}

// Matrix multiply kernel that uses shared memory for A
__global__ void mat_mult_tile_shared_AB_vector (
                        float_type* A_star, 
                        float_type* B_star, 
                        float_type* C,
                        // number of elements in a chunk
                        size_t chunk_len,
                        // number of chunks
                        size_t nchunks,
                        // length of a vector
                        size_t vector_len,
                        // number of vectors in a chunk
                        size_t nvectors,
                        // dimension 0 extent of C
                        size_t N0_C,
                        // dimension 1 extent of C
                        size_t N1_C) { 
    
    // Access the allocation of shared memory
    extern __shared__ char shared[];
    
    // N1_A_star >= N1_A
    // N1_A_star = nchunks * chunk_len
    // A_star is of size (N0_C, N1_A_star)
    // B_star is of size (N1_A_star, N1_C)
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Location within the block
    size_t s0=threadIdx.y;
    size_t s1=threadIdx.x;
    
    // block size
    size_t L0=blockDim.y;
    size_t L1=blockDim.x;

    // Get a pointer to shared_A from shared
    // shared_A is of size (L0, chunk_len)
    // shared_B is of size (L1, chunk_len)
    
    float_type* shared_A = (float_type*)&shared[0];
    float_type* shared_B = (float_type*)&shared[L0*chunk_len*sizeof(float_type)];

    // Line of shared memory 
    float_type* shared_A_star_s0 = &shared_A[s0*chunk_len];
    float_type* shared_B_star_s1 = &shared_B[s1*chunk_len];    
    
    // Line of shared memory as a vector
    float_vector_type* shared_A_star_v0 = (float_vector_type*)shared_A_star_s0;
    float_vector_type* shared_B_star_v1 = (float_vector_type*)shared_B_star_s1;
    
    // Scratch variable
    float_vector_type temp=(float_vector_type){0.0f};
    
    // Start and end positions to copy within a chunk
    size_t start0, end0, start1, end1;
    get_start_end(L1, chunk_len, s1, &start1, &end1);
    get_start_end(L0, chunk_len, s0, &start0, &end0);
    
    // Make sure we don't go beyond the bounds of the array
    if ((i0<N0_C) && (i1<N1_C)) {      
    
        // Loop over the chunks
        for (int chunk_id=0; chunk_id<nchunks; chunk_id++) {
    
            // Fill shared_A_star and shared_B_star 
            // Starting positions for the copy
            float_type* A_star_i0 = &A_star[i0*chunk_len*nchunks+chunk_id*chunk_len];
            float_type* B_star_i1 = &B_star[chunk_id*chunk_len*N1_C+i1];
        
            // Fill the rows of shared_A_star and shared_B_star
            // Copy from row i0 of A_star
            for (size_t n = start1; n<end1; n++) {
                shared_A_star_s0[n] = A_star_i0[n];
            }
        
            // Copy from column i1 of B_star   
            for (size_t n = start0; n<end0; n++) {
                shared_B_star_s1[n] = B_star_i1[n*N1_C];
            }
        
            // Synchronise threads to ensure shared memory is filled
            __syncthreads();
        
            // Loop over shared memory to compute dot product 
            // component for the chunk
            for (size_t n=0; n<nvectors; n++) {
                
                // Perform the dot product using shared memory           
#ifdef __HIP_PLATFORM_NVIDIA__
                temp.x += shared_A_star_v0[n].x*shared_B_star_v1[n].x;
                temp.y += shared_A_star_v0[n].y*shared_B_star_v1[n].y;
                temp.z += shared_A_star_v0[n].z*shared_B_star_v1[n].z;
                temp.w += shared_A_star_v0[n].w*shared_B_star_v1[n].w;
#else
                temp+=shared_A_star_v0[n]*shared_B_star_v1[n];
#endif
                 
            }
        
            // Synchronise threads so they are
            // are ready to tackle the next tile together
            __syncthreads();
        }
    
        // Put the accumulated value into position
        C[i0*N1_C+i1]=temp.x+temp.y+temp.z+temp.w;
    }
}

// Function to decide how much shared memory to allocate
void prep_kernel(
    const void *kernel, // The kernel function 
    void** kernel_args, // Arguments to the kernel
    dim3 num_blocks, // Number of blocks along dimensions of the grid
    dim3 block_size, // Number of threads along dimensions of the block
    size_t* sharedMemBytes, // Storage for shared memory
    void** prep_kernel_args // Arguments for this function
) {

    // Extract width of a chunk from prep kernel_args
    size_t* chunk_width = (size_t*)prep_kernel_args[0];

    // Set shared_bytes using line_width
    *sharedMemBytes=(*chunk_width)*(block_size.x+block_size.y);
}

int main(int argc, char** argv) {

    //// Step 1. Parse program arguments ////

    // Parse command line arguments
    int dev_index = h_parse_args(argc, argv);
    
    // Number of devices discovered
    int num_devices=0;
    
    //// Step 2. Discover resources and choose a compute device ////
    
    // Helper function to acquire devices
    // This sets the default device
    h_acquire_devices(&num_devices, dev_index);
        
    // Report on the device in use
    h_report_on_device(dev_index);
    
    // We are going to do a simple array multiplication for this example, 
    // using raw binary files for input and output
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)    
    // C is of size (N0_C, N1_C)

    size_t N1_A = NCOLS_A, N0_C = NROWS_C, N1_C = NCOLS_C;
    
    // Divide the N1_A axis into chunks
    // Use the alignment as the fundamental unit of a chunk length
    size_t chunk_len = h_get_alignment()/sizeof(float_type);
    
    // Vector length
    size_t vector_len = sizeof(float_vector_type)/sizeof(float_type);
    
    // Make sure an integer number of vectors fit into chunk_len 
    chunk_len = h_lcm(chunk_len, vector_len);
    
    size_t nchunks = N1_A/chunk_len;
    // Enlarge nchunks if there is not enough of them
    if (N1_A % chunk_len) nchunks++;
    
    // Number of vectors in a chunk
    size_t nvectors = chunk_len / vector_len;
    
    // Size of enlarged arrays
    size_t N1_A_star = nchunks*chunk_len;
    size_t nbytes_A_star = N0_C*N1_A_star*sizeof(float_type);
    size_t nbytes_B_star = N1_C*N1_A_star*sizeof(float_type);
    
    // A_star is of size (N0_C, N1_A_star)
    // B_star is of size (N1_A_star, N1_C)  
    
    //// Step 3. Construct matrices A_h and B_h on the host 
    //// and fill them with random numbers ////
    
    // Number of bytes in each array
    size_t nbytes_A = N0_C*N1_A*sizeof(float_type);
    size_t nbytes_B = N1_A*N1_C*sizeof(float_type);
    size_t nbytes_C = N0_C*N1_C*sizeof(float_type);

    // Allocate memory for the host arrays
    float* A_h = (float_type*)h_alloc(nbytes_A);
    float* B_h = (float_type*)h_alloc(nbytes_B);
    float* C_h = (float_type*)h_alloc(nbytes_C);

    // Fill the host arrays with random numbers 
    // using the matrix helper library
    m_random(A_h, N0_C, N1_A);
    m_random(B_h, N1_A, N1_C);
    
    //// Step 4. Allocate memory for arrays //// 
    //// A_star_d, B_star_d, and C_d on the compute device ////

    float *A_star_d, *B_star_d, *C_d;
    H_ERRCHK(hipMalloc((void**)&A_star_d, nbytes_A_star));
    H_ERRCHK(hipMalloc((void**)&B_star_d, nbytes_B_star));
    H_ERRCHK(hipMalloc((void**)&C_d, nbytes_C));

    // Zero out A_star_d, B_star_d
    H_ERRCHK(hipMemset(A_star_d, 0, nbytes_A_star));
    H_ERRCHK(hipMemset(B_star_d, 0, nbytes_B_star));

    //// Step 5. 1. Upload matrices A_h and B_h from the host //// 
    //// to enlarged arrays A_star_d and B_star_d on the device ////

    // A_star is of size (N0_C, N1_A_star)
    // B_star is of size (N1_A_star, N1_C) 
    
    H_ERRCHK(
        hipMemcpy2D(
            A_star_d,
            N1_A_star*sizeof(float_type),
            A_h,
            N1_A*sizeof(float_type),
            N1_A*sizeof(float_type),
            N0_C,
            hipMemcpyHostToDevice
        )
    );
    H_ERRCHK(
        hipMemcpy2D(
            B_star_d,
            N1_C*sizeof(float_type),
            B_h,
            N1_C*sizeof(float_type),
            N1_C*sizeof(float_type),
            N1_A,
            hipMemcpyHostToDevice
        )
    );        
 
    //// Step 6. Run the kernel to compute C_d ///
    //// from A_d and B_d on the device ////
        
    // Desired block size
    dim3 block_size = { 16, 16, 1 };
    dim3 global_size = { (uint32_t)N1_C, (uint32_t)N0_C, 1 };
    
    // Arguments for prep_kernel
    size_t chunk_width = chunk_len*sizeof(float_type); // bytes
    void* prep_kernel_args[] = { &chunk_width };
    
    // Arguments for the kernel
    void* kernel_args[] = { &A_star_d, &B_star_d, &C_d, 
                           &chunk_len, &nchunks, 
                           &vector_len, &nvectors,
                           &N0_C, &N1_C };
    
    // Find the optimum block size
    h_optimise_block(
        argc, // Number of command line arguments
        argv, // Command line arguments as an array of C-strings
        (const void*)&mat_mult_tile_shared_AB_vector, // Kernel function to execute
        kernel_args, // Arguments passed to the kernel  
        global_size, // Desired global_size
        &block_size, // Default block size
        (size_t)NSTATS, // Number of statistical runs per experiment
        0.0, // No prior times required
        prep_kernel, // No function required to prep the kernel
        prep_kernel_args // No arguments to prep function
    );
    
    // Wait for any commands to complete on the compute device
    H_ERRCHK(hipDeviceSynchronize());

    //// Step 7. Copy the buffer for matrix C_d //// 
    //// on the device back to C_h on the host ////
    H_ERRCHK(hipMemcpy((void*)C_h, (const void*)C_d, nbytes_C, hipMemcpyDeviceToHost));
    
    //// Step 8. Test the computed matrix **C_h** against a known answer
    
    // Compute the serial solution using the matrix helper library
    float* C_answer_h = (float*)calloc(nbytes_C, 1);
    m_mat_mult(A_h, B_h, C_answer_h, N1_A, N0_C, N1_C);
    
    // Uncomment this to check against elementwise matrix multiplication
    // m_hadamard(A_h, B_h, C_answer_h, N0_C, N1_C);

    // Print the maximum error between matrices
    float max_err = m_max_error(C_h, C_answer_h, N0_C, N1_C);
    
    //// Step 9. Write the contents of matrices A_h, B_h, and C_h to disk ////

    // Write out the host arrays to file
    h_write_binary(A_h, "array_A.dat", nbytes_A);
    h_write_binary(B_h, "array_B.dat", nbytes_B);
    h_write_binary(C_h, "array_C.dat", nbytes_C);
    
    //// Step 10. Clean up memory alllocations and release resources
    
    // Free the HIP buffers
    H_ERRCHK(hipFree(A_star_d));
    H_ERRCHK(hipFree(B_star_d));
    H_ERRCHK(hipFree(C_d));

    // Clean up host memory
    free(A_h);
    free(B_h);
    free(C_h);

    // Free the answer matrix
    free(C_answer_h);
    
    // Reset compute devices
    h_reset_devices(num_devices);
}

