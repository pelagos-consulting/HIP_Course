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

// Matrix multiply kernel that uses shared memory for A
__global__ void mat_mult_tile_shared_B (
                        float_type* A_star, 
                        float_type* B_star, 
                        float_type* C,
                        size_t chunk_len,
                        size_t nchunks, 
                        size_t N0_C,
                        size_t N1_C) { 

    // Access the allocation of shared memory
    extern __shared__ char shared[];
    
    // N1_A_star >= N1_A
    // A_star is of size (N0_C, N1_A_star)
    // B_star is of size (N1_A_star, N1_C)
    // C is of size (N0_C, N1_C)
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We assume row-major ordering for the matrices 
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;

    // Number of elements along N1_A
    size_t N1_A_star = nchunks*chunk_len;
    
    // Location within the block
    int s0=threadIdx.y;
    int s1=threadIdx.x;
    
    // block size
    int L0=blockDim.y;
    int L1=blockDim.x;

    // Get pointers to shared memory from shared
    // shared_B is of size (L1, chunk_len)
    float_type* shared_B = (float_type*)&shared[0];
    
    // Scratch variable
    float_type temp=(float_type)0.0;

    // Index of the thread within the workgroup, and total number of threads
    int w0 = s0*L1 + s1;
    int nthreads = L0*L1;

    // Make sure we don't go beyond the bounds of the array
    if ((i0<N0_C) && (i1<N1_C)) {

        // Make sure we don't go off the deep end of matrix B
        size_t max_offset=N1_C*N1_A_star;
        
        // Loop over the chunks
        for (int chunk_id=0; chunk_id<nchunks; chunk_id++) {

            // Fill shared_B using all threads
            for (int offset_S=w0; offset_S<L1*chunk_len; offset_S+=nthreads) {
    
                // Coordinates within shared_B of size (L1, chunk_len)
                int j0 = offset_S / chunk_len;
                int j1 = offset_S % chunk_len;
    
                // Get the offset into B of size (N1_A_star, N1_C)
                size_t offset_B = (chunk_id*chunk_len+j1)*N1_C 
                    + blockIdx.x*blockDim.x+j0;
                    
                if (offset_B<max_offset) {
                    shared_B[offset_S] = B_star[offset_B];
                } else {
                    shared_B[offset_S] = 0.0;
                }
            }
            
            // Synchronise threads to ensure shared memory is filled
            __syncthreads();
    
            // Get a scalar handle to shared memory
            float_type* shared_B_s = &shared_B[s1*chunk_len];
            
            // Loop over elements of shared memory and accumulate the dot product
            // component for the chunk
            for (int n=0; n<chunk_len; n++) {    
                // Perform the dot product using shared memory
                temp+=A_star[i0*N1_A_star+chunk_id*chunk_len+n]*shared_B_s[n];
            }
            
            // Synchronise threads so they are
            // are ready to tackle the next tile together
            __syncthreads();
        }

        // Put the accumulated value into position
        C[i0*N1_C+i1]=temp;
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
    *sharedMemBytes=(*chunk_width)*(block_size.x);
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
    size_t nchunks = N1_A/chunk_len;
    // Enlarge nchunks if there is not enough of them
    if (N1_A % chunk_len) nchunks++;
    
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
                           &chunk_len, &nchunks, &N0_C, &N1_C };
    
    // Find the optimum block size
    h_optimise_block(
        argc, // Number of command line arguments
        argv, // Command line arguments as an array of C-strings
        (const void*)&mat_mult_tile_shared_B, // Kernel function to execute
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

