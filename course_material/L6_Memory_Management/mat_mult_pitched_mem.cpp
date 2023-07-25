/* Code to perform a Matrix multiplication using HIP
Written by Dr Toby M. Potter
*/

// Setup headers
#include <cassert>
#include <cmath>
#include <iostream>

// Bring in the size of the matrices
#include "mat_size.hpp"

// Bring in a library to manage matrices on the CPU
#include "mat_helper.hpp"

// Bring in helper header to manage boilerplate code
#include "hip_helper.hpp"

// standard matrix multiply kernel 
__global__ void mat_mult (
        float* A, 
        float* B, 
        float* C,
        size_t pitch_A, // The pitch of A (in bytes)
        size_t N1_A, 
        size_t N0_C,
        size_t N1_C) { 
    
    // pitch_A_N is the pitch of A in elements
    size_t pitch_A_N = pitch_A/sizeof(float);
    
    // A is of size (N0_C, N1_A) with a pitch of pitch_A
    // B is of size (N1_A, N1_C)
    // C is of size (N0_C, N1_C)   
    
    // i0 and i1 represent the coordinates in Matrix C 
    // We use row-major ordering for the matrices
    
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Scratch variable
    float temp=0.0f; 

    // Guard mechanism to make sure we do not go
    // outside the boundaries of matrix C 
    if ((i0<N0_C) && (i1<N1_C)) {
        // Get the offset within the memory allocation of C
        size_t offset = i0*N1_C+i1;
        
        // Loop over columns of A and rows of B
        for (size_t n=0; n<N1_A; n++) {
            
            // A is of size (N0_C, N1_A) with a pitch of pitch_A_N (in elements) 
            // B is of size (N1_A, N1_C)
            
            // Loop across row i0 of A
            // and down column i1 of B
            temp+=A[i0*pitch_A_N+n]*B[i1+n*N1_C]; 
        }
        
        // Set the value in C at offset
        C[offset]=temp;
    }
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

    //// Step 3. Construct matrices A_h and B_h on the host 
    //// and fill them with random numbers ////
    
    // Number of bytes in each array
    size_t nbytes_A = N0_C*N1_A*sizeof(float);
    size_t nbytes_B = N1_A*N1_C*sizeof(float);
    size_t nbytes_C = N0_C*N1_C*sizeof(float);

    // Allocate memory for the host arrays
    float* A_h = (float*)h_alloc(nbytes_A);
    float* B_h = (float*)h_alloc(nbytes_B);
    float* C_h = (float*)h_alloc(nbytes_C);

    // Fill the host arrays with random numbers 
    // using the matrix helper library
    m_random(A_h, N0_C, N1_A);
    m_random(B_h, N1_A, N1_C);
    
    //// Step 4. Allocate memory for arrays //// 
    //// A_d, B_d, and C_d on the compute device ////

    float *A_d, *B_d, *C_d;
    size_t pitch_A;
    
    // Allocate A using pitched memory
    H_ERRCHK(
        hipMallocPitch(
            (void**)&A_d,
            &pitch_A, // actual given width of pencils
            N1_A*sizeof(float), // requested pencil width (bytes)
            N0_C // Height is the number of pencils to allocate for
        )
    );
    
    // Initialise pitched memory with a value
    H_ERRCHK(
        hipMemset(
            A_d, // The pointer to set
            0, // The value to fill
            pitch_A*N0_C // Number of bytes to fill
        )
    );
    
    // Allocate memory normally for B_d and C_d
    H_ERRCHK(hipMalloc((void**)&B_d, nbytes_B));
    H_ERRCHK(hipMalloc((void**)&C_d, nbytes_C));

    //// Step 5. 1. Upload matrices A_h and B_h from the host //// 
    //// to A_d and B_d on the device ////
    H_ERRCHK(
        hipMemcpy2D(
            A_d, // destination pointer
            pitch_A, // destination pitch
            A_h, // source pointer
            N1_A*sizeof(float), // source pitch
            N1_A*sizeof(float), // bytes along a pencil to transfer
            N0_C, // height (number of pencils to transfer)
            hipMemcpyHostToDevice // copy flag
        )
    );
        
    // Copy memory from B_h to B_d
    H_ERRCHK(hipMemcpy(B_d, B_h, nbytes_B, hipMemcpyHostToDevice));
 
    //// Step 6. Run the kernel to compute C_d ///
    //// from A_d and B_d on the device ////
        
    // Desired block size
    dim3 block_size = { 8, 8, 1 };
    dim3 global_size = { (uint32_t)N1_C, (uint32_t)N0_C, 1 };
    dim3 grid_nblocks;
    
    // Choose the number of blocks so that grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);

    // Amount of shared memory to use in the kernel
    size_t sharedMemBytes=0;
    
    // Launch the kernel using hipLaunchKernelGGL method
    // Use 0 when choosing the default (null) stream
    hipLaunchKernelGGL(mat_mult, 
            grid_nblocks, 
            block_size, sharedMemBytes, 0, 
            A_d, B_d, C_d,
            pitch_A,
            N1_A,
            N0_C,
            N1_C
    );
    
    // Alternatively, launch the kernel using CUDA triple Chevron syntax
    // which is not valid ANSI C++ syntax
    //mat_mult<<<grid_nblocks, block_size, 0, 0>>>(A_d, B_d, C_d, pitch_A, N1_A, N0_C, N1_C);
    
    // Wait for any commands to complete on the compute device
    H_ERRCHK(hipDeviceSynchronize());

    //// Step 7. Copy the buffer for matrix C_d //// 
    //// on the device back to C_h on the host ////
    
    // Create the object to hold 
    // the parameters of the 3D copy
    hipMemcpy3DParms copy_parms = {0};
    
    // Use hipMemcpy3D for reference
    
    // Create pitched pointers 
    // For the host
    hipPitchedPtr C_h_ptr = make_hipPitchedPtr(
        C_h, // pointer 
        N1_C*sizeof(float), // pitch - actual pencil width (bytes) 
        N1_C, // requested pencil width (elements)
        N0_C // number of pencils in a plane (elements)
    );
    // For the device
    hipPitchedPtr C_d_ptr = make_hipPitchedPtr(
        C_d, // pointer
        N1_C*sizeof(float), // pitch - actual pencil width (bytes) 
        N1_C, // requested pencil width (elements)
        N0_C // number of pencils in a plane (elements)
    );
    // Postion within the host array
    hipPos C_h_pos = make_hipPos(
        0*sizeof(float), // byte position along a pencil (bytes)
        0, // starting pencil index (elements)
        0 // start pencil plane index (elements)
    );
    // Postion within the device array
    hipPos C_d_pos = make_hipPos(
        0*sizeof(float), // byte position along a pencil (bytes)
        0, // starting pencil index (elements)
        0 // starting pencil plane index (elements)
    );
    // Choose the region to copy
    hipExtent extent = make_hipExtent(
        N1_C*sizeof(float), // width of pencil region to copy (bytes)
        N0_C, // number of pencils to copy the region from
        1 // number of pencil planes
    );
    
    // Fill the copy parameters
    copy_parms.srcPtr = C_d_ptr;
    copy_parms.srcPos = C_d_pos;
    copy_parms.dstPtr = C_h_ptr;
    copy_parms.dstPos = C_h_pos;
    
    copy_parms.extent = extent;
    copy_parms.kind = hipMemcpyDeviceToHost;
    
    H_ERRCHK(hipMemcpy3D(&copy_parms));
    
    //// Step 8. Test the computed matrix **C_h** against a known answer
    
    // Compute the serial solution using the matrix helper library
    float* C_answer_h = (float*)calloc(nbytes_C, 1);
    m_mat_mult(A_h, B_h, C_answer_h, N1_A, N0_C, N1_C);
    
    // Uncomment this to check against elementwise matrix multiplication
    // m_hadamard(A_h, B_h, C_answer_h, N0_C, N1_C);

    // Print the maximum error between matrices
    float max_err = m_max_error(C_h, C_answer_h, N0_C, N1_C);
    
    //// Step 9. Write the contents of matrices 
    //// A_h, B_h, and C_h to disk

    // Write out the host arrays to file
    h_write_binary(A_h, "array_A.dat", nbytes_A);
    h_write_binary(B_h, "array_B.dat", nbytes_B);
    h_write_binary(C_h, "array_C.dat", nbytes_C);
    
    //// Step 10. Clean up memory alllocations and release resources
    
    // Free the HIP buffers
    H_ERRCHK(hipFree(A_d));
    H_ERRCHK(hipFree(B_d));
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

