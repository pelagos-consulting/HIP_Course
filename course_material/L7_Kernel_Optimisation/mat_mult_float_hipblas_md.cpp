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

// Bring in the hipblas header files
#include <hipblas/hipblas.h>

typedef float float_type;

int main(int argc, char** argv) {
    
    //// Step 1. Parse program arguments and initialise hipblas ////
    
    // Parse command line arguments
    int dev_index = h_parse_args(argc, argv);
    
    // Number of devices discovered
    int num_devices=0;
    
    //// Step 2. Discover resources and choose a compute device ////
    
    // Helper function to acquire devices
    // This sets the default device
    h_acquire_devices(&num_devices, dev_index);
    
    // Initialize hipblas
    hipblasHandle_t hb_handle;
    hipblasStatus_t hb_status = hipblasCreate(&hb_handle);
    if (hb_status != HIPBLAS_STATUS_SUCCESS) {
        std::printf("Failed to initialise hipBlas\n");
         exit(EXIT_FAILURE); 
    }
    
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
    size_t nbytes_A = N0_C*N1_A*sizeof(float_type);
    size_t nbytes_B = N1_A*N1_C*sizeof(float_type);
    size_t nbytes_C = N0_C*N1_C*sizeof(float_type);

    // Allocate memory for the host arrays
    float_type* A_h = (float_type*)h_alloc(nbytes_A);
    float_type* B_h = (float_type*)h_alloc(nbytes_B);
    float_type* C_h = (float_type*)h_alloc(nbytes_C);

    // Fill the host arrays with random numbers 
    // using the matrix helper library
    m_random(A_h, N0_C, N1_A);
    m_random(B_h, N1_A, N1_C);
    
    //// Step 4. Allocate memory for arrays //// 
    //// A_d, B_d, and C_d on the compute device ////

    //// How many elements are there in a slab
    size_t slab_len = 64;
    size_t nslabs = N0_C/slab_len;
    assert(N0_C%slab_len==0);
    size_t nbytes_A_slab = slab_len*N1_A*sizeof(float_type);
    size_t nbytes_C_slab = slab_len*N1_C*sizeof(float_type);

    // Arrays of device allocations for each device
    float_type** As_d = (float_type**)calloc((size_t)num_devices, sizeof(float_type*));
    float_type** As_slab_d = (float_type**)calloc((size_t)num_devices, sizeof(float_type*));
    float_type** Bs_d = (float_type**)calloc((size_t)num_devices, sizeof(float_type*));
    float_type** Cs_slab_d = (float_type**)calloc((size_t)num_devices, sizeof(float_type*));

    for (int i=0; i<num_devices; i++) {
        // Choose the device to use
        H_ERRCHK(hipSetDevice(i));
        H_ERRCHK(hipMalloc((void**)&As_d[i], nbytes_A));
        H_ERRCHK(hipMalloc((void**)&Bs_d[i], nbytes_B));

        // Memory for a slab
        H_ERRCHK(hipMalloc((void**)&As_slab_d[i], nbytes_A_slab));
        H_ERRCHK(hipMalloc((void**)&Cs_slab_d[i], nbytes_C_slab));

        H_ERRCHK(hipMemcpy(As_d[s], A_h, nbytes_A, hipMemcpyHostToDevice));
        H_ERRCHK(hipMemcpy(Bs_d[i], B_h, nbytes_B, hipMemcpyHostToDevice));
    }
   
    //// Step 5. 1. Upload matrices A_h and B_h from the host //// 
    //// to A_d and B_d on the device ////
 
    //// Step 6. Run the kernel to compute C_d ///
    //// from A_d and B_d on the device ////

    // Setup memory for the experiment to run
    int nexperiments=1;
    int npoints=2;
    size_t nbytes_output = nexperiments*npoints*sizeof(double);
    double* output_local = (double*)malloc(nbytes_output);
    
    // Run the experiment nstats times
    const size_t nstats=NSTATS;
    double times_ms[nstats] = {0};
    double time_ms=0.0;
    double avg_time_ms=0.0;
    double max_time_ms=0.0;
    int max_time_n = 0;
    
    // Run the hipblas Sgemm routine nstats times and collect times
    for (int n=0; n<nstats; n++) {
    
        // Start the clock
        auto t1 = std::chrono::high_resolution_clock::now();
    
        // Call the hipblas Sgemm routine to perform the matrix multiplication
        // for single precision matrices
    
        const float_type alpha = 1.0f;
        const float_type beta = 0.0f;
        
        // Now we try to compute
        // AB = C
        // But hipblasSgemm computes
        // alpha* op( A )*op( B ) + beta*C = C
        // where A, B, and C are column major
    
        // We want to input row_major matrices, we know that
    
        // A_{row_major} = A^{T}_{col_major}
        // B_{row_major} = B^{T}_{col_major}
        // C_{row_major} = C^{T}_{col_major}
    
        // This identity is useful
        // (AB)^T = B^T A^T = C^T
    
        // Then 
    
        // B^{T}_{col_major} A^{T}_{col_major} = C^{T}_{col_major}
        // Is equivalent to....
        // B_{row_major} A_{row_major} = C_{row_major}
     
        // So we replace in the call to hipblasSgemm
    
        // A_{col_major} -> B_{row_major}
        // B_{col_major} -> A_{row_major}
        // m -> n, n -> m
    
        // Leading dimension is the length along dimension of matrix 
        // along which memory is contiguous
        // for col_major arrays this is dimension 0

        // Iterate over slabs
        #pragma omp parallel for shared(command_queues, As_d, Bs_d, Cs_slab_d, C_h, N0_C, N1_C, kernel_times) default(none) schedule(dynamic,1)  
        for (int s=0; s<nslabs; s++) {

            // Get the thread ID
            int tid = omp_get_thread_num();
            H_ERRCHK(hipSetDevice(tid));

            // Copy memory to As_slab_d from As_d
            // GOT TO HERE

            hb_status = hipblasSgemm(
                hb_handle, // hipblas handle  
                HIPBLAS_OP_N, // what operation to apply to A (none)
                HIPBLAS_OP_N, // what operation to apply to B (none)
                (int)N1_C, // number of rows in C_{col_major} -> number of columns in C_{row_major}
                (int)N0_C, // number of columns in C_{col_major} -> number of rows in C_{row_major}
                (int)N1_A, // number of columns in A_{col_major} -> number of rows in B_{row_major}
                &alpha, // Constant
                B_d, // Normally A_{col_major} -> B^{T}_{col_major} -> B_{row_major}
                (int)N1_C, // Leading dimension for B_{row_major} 
                A_d, // Normally B_{col_major} -> A^{T}_{col_major} -> A_{row_major}
                (int)N1_A, // Leading dimension for A_{row_major}
                &beta, // Constant
                C_d, // Pointer to memory for C_{row_major}
                (int)N1_C // Leading dimension for C_{row_major}
            );
    
            if (hb_status != HIPBLAS_STATUS_SUCCESS) {
                std::printf("Failed to run hipBlas function.\n");
                exit(EXIT_FAILURE); 
            }
    
            // Wait for any commands to complete on the compute device
            H_ERRCHK(hipDeviceSynchronize());

            // Copy portion of memory back to C_h
            // GOT TO HERE

        }

        // Stop the clock
        auto t2 = std::chrono::high_resolution_clock::now();
        
        // Get the time
        time_ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000.0;
        
        // Keep track of the maximum time
        if (time_ms > max_time_ms) {
            max_time_ms = time_ms;
            max_time_n = n;
        }
        
        times_ms[n]=time_ms;
        
        avg_time_ms+=time_ms;
    }           
        
    //// Step 7. Copy the buffer for matrix C_d //// 
    //// on the device back to C_h on the host ////
    H_ERRCHK(hipMemcpy((void*)C_h, (const void*)C_d, nbytes_C, hipMemcpyDeviceToHost));
    
    //// Step 8. Test the computed matrix **C_h** against a known answer
    
    // Compute the serial solution using the matrix helper library
    float_type* C_answer_h = (float_type*)calloc(nbytes_C, 1);
    m_mat_mult(A_h, B_h, C_answer_h, N1_A, N0_C, N1_C);
    
    // Uncomment this to check against elementwise matrix multiplication
    // m_hadamard(A_h, B_h, C_answer_h, N0_C, N1_C);

    // Print the maximum error between matrices
    float_type max_err = m_max_error(C_h, C_answer_h, N0_C, N1_C);
    
    //// Step 9. Write the contents of matrices A_h, B_h, and C_h to disk ////

    // Write out the host arrays to file
    h_write_binary(A_h, "array_A.dat", nbytes_A);
    h_write_binary(B_h, "array_B.dat", nbytes_B);
    h_write_binary(C_h, "array_C.dat", nbytes_C);
    
    // Calculate the mean and average times
    // Leave the longest time out of the calculation
    avg_time_ms = avg_time_ms - max_time_ms;
    avg_time_ms/=(double)(nstats-1);
    double std_time_ms=0.0, scratch=0.0;
    
    for (int n=0; n<nstats; n++) {
        scratch=times_ms[n]-avg_time_ms;
        if (n!=max_time_n) {
            std_time_ms+=(scratch*scratch);
        }
    }
    std_time_ms=sqrt(std_time_ms)/(double)(nstats-1);
    
    output_local[0]=avg_time_ms;
    output_local[1]=std_time_ms;
    
    h_write_binary(output_local, "output_block.dat", nbytes_output);
    free(output_local);
    
    //// Step 10. Clean up memory alllocations and release resources
    
    // Free the HIP buffers
    for (int i=0; i<num_devices; i++) {
        // Choose the device to use
        H_ERRCHK(hipSetDevice(i));
        H_ERRCHK(hipFree(As_d[i]));
        H_ERRCHK(hipFree(Bs_d[i]));

        // Memory for a slab
        H_ERRCHK(hipFree(As_slab_d[i]));
        H_ERRCHK(hipFree(Cs_slab_d[i]));
    }

    // Clean up host memory
    free(A_h);
    free(B_h);
    free(C_h);
    
    free(As_d);
    free(Bs_d);
    free(As_slab_d);
    free(Bs_slab_d);

    // Free the answer matrix
    free(C_answer_h);
    
    // Uninitialise hipblas
    hipblasDestroy(hb_handle);
    
    // Reset compute devices
    h_reset_devices(num_devices);
}

