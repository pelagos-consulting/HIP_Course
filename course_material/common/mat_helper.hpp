///
/// @file  mat_helper.hpp
/// 
/// @brief Matrix helper functions.
///
/// Written by Dr. Toby Potter 
/// for the Commonwealth Scientific and Industrial Research Organisation of Australia (CSIRO).
///

#include <cmath>
#include <cstdlib>
#include <random>
#include <ctime>
#include <iostream>
#include <iomanip>

/// Fill a matrix with random numbers
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

/// Pretty print out a matrix
template<typename T>
void m_show_matrix(T* src, size_t N0, size_t N1) {

    // Set precision
    std::cout.precision(2);
    std::cout << std::scientific;

    for (size_t i1=0; i1<N1; i1++) {
        std::cout << "-";
    }

    std::cout << "\n";

    // Pretty print the matrix
    for (size_t i0=0; i0<N0; i0++) {

        for (size_t i1=0; i1<N1; i1++) {
            size_t offset = i0*N1 + i1;

            if (i1==0) {
                std::cout << "|";
            }

            std::cout << " " << std::setw(9) << src[offset];

            if (i1 == N1-1) {
                std::cout << " |\n";
            }
        }
    }

    for (size_t i1=0; i1<N1; i1++) {
        std::cout << "-";
    }

    // Reset formatting
    std::cout << std::defaultfloat;
    std::cout << "\n";
}

/// Do matrix multiplication on the CPU
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

/// Do Hadamard (elementwise) matrix multiplication
template<typename T>
void m_hadamard(T* A, T* B, T* C, size_t N0_C, size_t N1_C) {
    
    for (size_t i0=0; i0<N0_C; i0++) {
        for (size_t i1=0; i1<N1_C; i1++) {
            // Temporary value
            size_t offset = i0*N1_C+i1;

            // Set the value in C
            C[offset] = A[offset]*B[offset];
        }
    }
}

/// Find the maximum absolute difference between two matrices
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

/// Find the difference between two matrices
template<typename T>
void m_residual(T* A, T* B, T* C, size_t N0, size_t N1) {
    // Function to compute the residual bewteen two matrices
    for (size_t i0=0; i0<N0; i0++) {
        for (size_t i1=0; i1<N1; i1++) {
            size_t offset = i0*N1 + i1;
            C[offset] = A[offset]-B[offset];
        }
    }
}

/// Cross-correlation on the CPU
template<typename T>
void m_xcorr(
        T* dst, 
        T* src, 
        T* krn, 
        size_t len0_src,
        size_t len1_src, 
        size_t pad0_l,
        size_t pad0_r,
        size_t pad1_l,
        size_t pad1_r) {

    // Assuming row-major ordering
    
    // Size of the kernel
    size_t K0=pad0_l+pad0_r+1;
    size_t K1=pad1_l+pad1_r+1;
    
    // Make sure the sizes work ok
    assert(len0_src>=K0);
    assert(len1_src>=K1);    
    
    // Compute a naive cross correlation on the CPU
    for (size_t i0=pad0_l; i0<len0_src-pad0_r; i0++) {
        for (size_t i1=pad1_l; i1<len1_src-pad1_r; i1++) {
            // Loop over the kernel
            T sum=0.0;
            for (size_t k0=0; k0<K0; k0++) {
                for (size_t k1=0; k1<K1; k1++) {
                    sum+=krn[k0*K1+k1]
                        *src[(i0-pad0_l+k0)*len1_src+(i1-pad1_l+k1)];
                }
            }
            
            // Put the matrix into the destination
            dst[i0*len1_src+i1]=sum;
        }
    }
}