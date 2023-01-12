#include "hip_helper.hpp"
#include "mat_size.hpp"
#include "mat_helper.hpp"

int main(int argc, char** argv) {
    
    // Read in the array
    size_t N1_A = NCOLS_A, N0_C = NROWS_C, N1_C = NCOLS_C;
    size_t nbytes_A, nbytes_B, nbytes_C;

    float* A_h = (float*)h_read_binary("array_A.dat", &nbytes_A);
    float* B_h = (float*)h_read_binary("array_B.dat", &nbytes_B);

    // Sanity check on incoming data
    assert(nbytes_A==N0_C*N1_A*sizeof(float));   
    assert(nbytes_B==N1_A*N1_C*sizeof(float));
    nbytes_C=N0_C*N1_C*sizeof(float);
    
    // Make an array on the host to store the result (matrix C)
    float* C_h = (float*)calloc(nbytes_C, 1);

    // Run the serial matrix multiplication on the CPU
    m_mat_mult(A_h, B_h, C_h, N1_A, N0_C, N1_C);

    // Write out the result to file
    h_write_binary(C_h, "array_C.dat", nbytes_C);

    // Free matrices
    free(A_h);
    free(B_h);
    free(C_h);
}
