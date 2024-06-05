/* Code to perform a Matrix multiplication using HIP
Written by Dr Toby M. Potter from Pelagos Consulting and Education
*/

// Setup headers
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

// Windows specific header instructions
#if defined(_WIN32) || defined(_WIN64)
    #define NOMINMAX
    #include <windows.h>
    #include <malloc.h>
#endif

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <cassert>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdint>

// Import the HIP header
#include <hip/hip_runtime.h>

// Align all memory allocations to this byte boundary
#define BYTE_ALIGNMENT 64

// Given a block_size, fit grid_nblocks to a desired global_size
void h_fit_blocks(dim3* grid_nblocks, dim3 global_size, dim3 block_size) {
    // Make grid_blocks big enough to fit a grid of at least global_size
    // when blocks are of size block_size     
    assert ((global_size.x>0) && (block_size.x>0));
    assert ((global_size.y>0) && (block_size.y>0));
    assert ((global_size.z>0) && (block_size.z>0));

    // Make the number of blocks
    (*grid_nblocks).x = global_size.x/block_size.x;
    if ((global_size.x % block_size.x)>0) {
        (*grid_nblocks).x += 1;
    }

    (*grid_nblocks).y = global_size.y/block_size.y;
    if ((global_size.y % block_size.y)>0) { 
        (*grid_nblocks).y += 1;
    }

    (*grid_nblocks).z = global_size.z/block_size.z;
    if ((global_size.z % block_size.z)>0) {
        (*grid_nblocks).z += 1;
    }
}

// Function to check error codes
void h_errchk(hipError_t errcode, const char* message) {
    // Function to check the error code
    if (errcode != hipSuccess) { 
        const char* errstring = hipGetErrorString(errcode); 
        std::fprintf( 
            stderr, 
            "Error, HIP call failed at %s, error string is: %s\n", 
            message, 
            errstring 
        ); 
        exit(EXIT_FAILURE); 
    }
}

// Macro to check error codes
#define H_ERRCHK(cmd) \
{\
    h_errchk(cmd, "__FILE__:__LINE__");\
}

void h_write_binary(void* data, const char* filename, size_t nbytes) {
    // Write binary data to file
    std::FILE *fp = std::fopen(filename, "wb");
    if (fp == NULL) {
        std::printf("Error in writing file %s", filename);
        exit(EXIT_FAILURE);
    }
    
    // Write the data to file
    std::fwrite(data, nbytes, 1, fp);
    
    // Close the file
    std::fclose(fp);
}

// Function to generate a matrix
template<typename T>
void m_random(T* dst, size_t N0, size_t N1) {

    // Initialise random number generator
    std::random_device rd;
    unsigned int seed = 100;
    // Non-deterministic random number generation
    //std::mt19937 gen(rd());
    // Deterministic random number generation
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(0,1);

    // Fill the array
    for (size_t i=0; i<N0*N1; i++) {
        dst[i]=(T)dist(gen);
    }
}

// Function to do matrix multiplication on the CPU
template<typename T>
void m_mat_mult(T* A, T* B, T* C, size_t N1_A, size_t N0_C, size_t N1_C) {
    
    for (size_t i0=0; i0<N0_C; i0++) {
        for (size_t i1=0; i1<N1_C; i1++) {
            // Temporary value
            T temp=0;

            // Offset to access arrays
            for (size_t n=0; n<N1_A; n++) {
                temp+=A[i0*N1_A+n]*B[n*N1_C+i1];
            }

            // Set the value in C
            C[i0*N1_C+i1] = temp;
        }
    }
}

// Function to find the maximum error between two matrices
template<typename T>
T m_max_error(T* M0, T* M1, size_t N0, size_t N1) {
    
    // Maximum error in a matrix
    T max_error = 0;

    for (size_t i0=0; i0<N0; i0++) {
        for (size_t i1=0; i1<N1; i1++) {
            size_t offset = i0*N1 + i1;
            max_error = std::fmax(max_error, std::fabs(M0[offset]-M1[offset]));
        }
    }

    std::cout << "Maximum error (infinity norm) is: " << max_error << "\n";
    return max_error;
}

// standard matrix multiply kernel 
__global__ void mat_mult (
        float* A, 
        float* B, 
        float* C, 
        size_t N1_A, 
        size_t N0_C,
        size_t N1_C) { 
            
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)
    // C is of size (N0_C, N1_C)   
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We use row-major ordering for the matrices
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Scratch variable
    float temp=0.0; 

    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Loop over columns of A and rows of B
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A)
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=A[i0*N1_A+n]*B[i1+n*N1_C]; 
        }
        
        // Set the value in C at offset
        C[i0*N1_C+i1]=temp;
   }
} 

int main(int argc, char** argv) {
    
    //// Step 1. Initialise HIP ////
    H_ERRCHK(hipInit(0));

    //// Step 2. Set the compute device ////
    int dev_index = 0;
    H_ERRCHK(hipSetDevice(dev_index));
    
    // We are going to do a simple array multiplication for this example, 
    
    // A is of size (N0_C, N1_A)
    // B is of size (N1_A, N1_C)    
    // C is of size (N0_C, N1_C)

    size_t N1_A = 72, N0_C = 72, N1_C = 72;

    //// Step 3. 1. Construct matrices A_h and B_h on the host 
    //// and fill them with random numbers ////
    
    // Number of bytes in each array
    size_t nbytes_A = N0_C*N1_A*sizeof(float);
    size_t nbytes_B = N1_A*N1_C*sizeof(float);
    size_t nbytes_C = N0_C*N1_C*sizeof(float);

    // Allocate pinned memory for the host arrays
    float *A_h, *B_h, *C_h;
    H_ERRCHK(hipHostMalloc((void**)&A_h, nbytes_A));
    H_ERRCHK(hipHostMalloc((void**)&B_h, nbytes_B));
    H_ERRCHK(hipHostMalloc((void**)&C_h, nbytes_C));

    // Fill the host arrays with random numbers 
    // using the matrix helper library
    m_random(A_h, N0_C, N1_A);
    m_random(B_h, N1_A, N1_C);
    
    //// Step 4. Allocate memory for arrays //// 
    //// A_d, B_d, and C_d on the compute device ////

    float *A_d, *B_d, *C_d;
    H_ERRCHK(hipMalloc((void**)&A_d, nbytes_A));
    H_ERRCHK(hipMalloc((void**)&B_d, nbytes_B));
    H_ERRCHK(hipMalloc((void**)&C_d, nbytes_C));

    //// Step 5. 1. Upload matrices A_h and B_h from the host //// 
    //// to A_d and B_d on the device ////
    H_ERRCHK(hipMemcpy(A_d, A_h, nbytes_A, hipMemcpyHostToDevice));
    H_ERRCHK(hipMemcpy(B_d, B_h, nbytes_B, hipMemcpyHostToDevice));
 
    //// Step 6. Run the kernel to compute C_d ///
    //// from A_d and B_d on the device ////
        
    // Desired block size
    dim3 block_size = { 16, 4, 1 };
    dim3 global_size = { (uint32_t)N1_C, (uint32_t)N0_C, 1 };
    dim3 grid_nblocks;
    
    // Choose the number of blocks so that Grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);

    // Amount of shared memory to use in the kernel
    size_t sharedMemBytes=0;
    
    // Launch the kernel using hipLaunchKernelGGL method
    hipLaunchKernelGGL(mat_mult, 
            grid_nblocks, 
            block_size, sharedMemBytes, 0, 
            A_d, B_d, C_d,
            N1_A,
            N0_C,
            N1_C
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
    
    // Print the maximum error between matrices
    float max_err = m_max_error(C_h, C_answer_h, N0_C, N1_C);
    
    //// Step 9. Write the contents of matrices A_h, B_h, and C_h to disk ////

    // Write out the host arrays to file
    h_write_binary(A_h, "array_A.dat", nbytes_A);
    h_write_binary(B_h, "array_B.dat", nbytes_B);
    h_write_binary(C_h, "array_C.dat", nbytes_C);
    
    //// Step 10. Clean up memory alllocations and release resources
    
    // Free the HIP buffers
    H_ERRCHK(hipFree(A_d));
    H_ERRCHK(hipFree(B_d));
    H_ERRCHK(hipFree(C_d));

    // Clean up pinned memory on the host   
    H_ERRCHK(hipHostFree(A_h));
    H_ERRCHK(hipHostFree(B_h));
    H_ERRCHK(hipHostFree(C_h));

    // Free the answer matrix
    free(C_answer_h);
    
    // Reset compute device
    H_ERRCHK(hipDeviceReset());
}

