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

// Kernel to perform the matrix multiplication
void mat_mult(
        float *A, 
        float *B, 
        float *C, 
        size_t N0_C, 
        size_t N1_C, 
        size_t N1_A, 
        size_t i0, 
        size_t i1) {

    // A is of size (N0_C, N1_A) = (4,3)
    // B is of size (N1_A, N1_C) = (3,6)   
    // C is of size (N0_C, N1_C) = (4,6)

    // Make sure we stay within the bounds of matrix C
    if ((i0<N0_C) && (i1<N1_C)) {
        
        // Define a loop constant
        float temp=0.0;

        // Loop N1_A elements along row i0 of A and down column i1 of B
        for (size_t n=0; n<N1_A; n++) {

            //// Exercise, fix these two wrong lines of code ////
            
            size_t offset_A = 0;
            size_t offset_B = 0;
            
            //// End exercise ////

            // Add to the temporary sum
            temp+=A[offset_A]*B[offset_B];
        }

        // Place temp into the matrix at coordinates (i0, i1)
        C[i0*N1_C+i1]=temp;
    }

}    

int main(int argc, char** argv) {

    // We are going to do a simple array multiplication for this example, 
        
    // NROWS_C, NCOLS_C, NCOLS_A are defined in mat_size.hpp
    size_t N0_C = NROWS_C, N1_C = NCOLS_C, N1_A=NCOLS_A;

    // Matrix A is of size (N0_C, N1_A) = (4,3)
    // Matrix B is of size (N1_A, N1_C) = (3,6)   
    // Matrix C is of size (N0_C, N1_C) = (4,6)

    //// Construct matrices A_h, B_h, and C_h  
    
    // Number of bytes in each array
    size_t nbytes_A = N0_C*N1_A*sizeof(float);
    size_t nbytes_B = N1_A*N1_C*sizeof(float);
    size_t nbytes_C = N0_C*N1_C*sizeof(float);

    // Allocate zeroed out memory for the arrays
    // The _h suffix means the memory is on the host
    float* A_h = (float*)calloc(nbytes_A, 1);
    float* B_h = (float*)calloc(nbytes_B, 1);
    float* C_h = (float*)calloc(nbytes_C, 1);

    // Fill matrices A_h and B_h with random numbers 
    // using a function m_random from the 
    // matrix helper library "mat_helper.hpp"
    m_random(A_h, N0_C, N1_A);
    m_random(B_h, N1_A, N1_C);
    
    // Visit every cell in matrix C and execute a kernel to compute a value there
    
    // Loop over the rows (i0) of C
   

    // Compute the serial solution using the matrix helper library
    float* C_answer_h = (float*)calloc(nbytes_C, 1);
    m_mat_mult(A_h, B_h, C_answer_h, N1_A, N0_C, N1_C);
    
    // Compute the residual between C_h and C_answer_h
    float* C_residual_h = (float*)calloc(nbytes_C, 1);
    m_residual(C_answer_h, C_h, C_residual_h, N0_C, N1_C);

    // Pretty print matrix C
    std::cout << "The computed array (C_h) is\n";
    m_show_matrix(C_h, N0_C, N1_C);

    std::cout << "The solution array (C_answer_h) is \n";
    m_show_matrix(C_answer_h, N0_C, N1_C);

    std::cout << "The residual (C_answer_h-C_h) is\n";
    m_show_matrix(C_residual_h, N0_C, N1_C);

    // Print the maximum error between matrices
    float max_err = m_max_error(C_h, C_answer_h, N0_C, N1_C);

    // Free the memory allocations
    free(A_h);
    free(B_h);
    free(C_h);
    free(C_answer_h);
    free(C_residual_h);
}

