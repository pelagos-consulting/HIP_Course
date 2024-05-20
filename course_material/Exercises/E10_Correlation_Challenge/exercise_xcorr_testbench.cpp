// Testbench for cross-correlation algorithm
#include <assert.h>
#include <cstdio>
#include <cstdint>
#include <omp.h>
#include <cmath>
#include <chrono>

// Include helper files
#include "hip_helper.hpp"
#include "mat_helper.hpp"

// Image sizes
#define N0 5
#define N1 4

// Amounts to pad
#define L0 0 
#define R0 2
#define L1 0
#define R1 2

typedef float float_type;

__global__ void xcorr(
        float_type *src, 
        float_type *dst,
        float_type *kern,
        size_t len0_src,
        size_t len1_src, 
        size_t pad0_l,
        size_t pad0_r,
        size_t pad1_l,
        size_t pad1_r      
    ) {

    // get the coordinates
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;

    //// Task 1 complete the correlation kernel ////
    
    // Uncomment for the shortcut answer
    // #include "task1_answer.hpp" 


    
    //// End Task 1 ////
}

int main(int argc, char** argv) {
    
    // Parse command line arguments
    int dev_index = h_parse_args(argc, argv);

    // Number of devices discovered
    int num_devices=0;

    // Report on the compute device
    h_report_on_device(dev_index);
    
    // Size of the kernel
    const size_t K0=L0+R0+1;
    const size_t K1=L1+R1+1;
    
    // Number of bytes for a single image
    size_t nbytes_image = N0*N1*sizeof(float_type);
    size_t nbytes_kern = K0*K1*sizeof(float_type);    
    
    // Allocate storage for the input and fill
    // with pseudorandom numbers
    float_type* image_in = (float_type*)h_alloc(nbytes_image); 
    m_random(image_in, N0, N1);
    
    // scale up random numbers for debugging purposes
    for (int i=0; i<N0*N1; i++) {
        image_in[i] = round(9.0*image_in[i]);
    }
    
    
    // Allocate storage for the output 
    float_type* image_out = (float_type*)h_alloc(nbytes_image);
 
    // Allocate storage for the test image
    float_type* image_test = (float_type*)h_alloc(nbytes_image);
    
    // Make the image kernel
    float_type image_kern[K0*K1] = {-1,-1,-1,\
                                -1, 8,-1,\
                                -1,-1,-1};

    // Allocate buffers for input image, output image, and kernel
    float *src_d, *dst_d, *krn_d;
    H_ERRCHK(hipMalloc((void**)&src_d, nbytes_image));
    H_ERRCHK(hipMalloc((void**)&dst_d, nbytes_image));
    H_ERRCHK(hipMalloc((void**)&krn_d, nbytes_kern));
   
    // Upload image kernel to device
    H_ERRCHK(
        hipMemcpy(
            krn_d, 
            image_kern,
            nbytes_kern, 
            hipMemcpyHostToDevice
        )
    );

    // Upload input image to the device
    H_ERRCHK(
        hipMemcpy(
            src_d, 
            image_in,
            nbytes_image, 
            hipMemcpyHostToDevice
        )
    );

    // Set memory in dst_d using hipMemset
    H_ERRCHK(hipMemset(dst_d, 0, nbytes_image));
    
    // Kernel size parameters
    dim3 block_size = { 8, 8, 1 };
    dim3 global_size = { (uint32_t)N1, (uint32_t)N0, 1 };
    dim3 grid_nblocks;

    // Choose the number of blocks so that grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);
        
    // Amount of shared memory to use in the kernel
    size_t sharedMemBytes=0;
            
    // Just for kernel arguments
    size_t len0_src = N0, len1_src = N1;
    size_t pad0_l = L0, pad0_r = R0, pad1_l = L1, pad1_r = R1;
            
    // Launch the kernel
    hipLaunchKernelGGL(
        xcorr, 
        grid_nblocks, 
        block_size, sharedMemBytes, 0, 
        src_d, dst_d, krn_d,
        len0_src,
        len1_src, 
        pad0_l,
        pad0_r,
        pad1_l,
        pad1_r   
    );
       
    // Copy memory back from the device
    H_ERRCHK(
        hipMemcpy(
            image_out, 
            dst_d, 
            nbytes_image, 
            hipMemcpyDeviceToHost
        )
    );

    // Perform the test correlation
    m_xcorr(
        image_test, 
        image_in, 
        image_kern,
        len0_src, len1_src, 
        pad0_l, pad0_r, 
        pad1_l, pad1_r
    );
    
    // Pretty print the matrices
    std::cout << "Image in" << "\n";
    m_show_matrix(image_in, len0_src, len1_src);    

    std::cout << "Image kernel" << "\n";
    m_show_matrix(image_kern, K0, K1); 
    
    std::cout << "Image out - CPU" << "\n";
    m_show_matrix(image_test, len0_src, len1_src);
    
    std::cout << "Image out - HIP" << "\n";
    m_show_matrix(image_out, len0_src, len1_src);
    
    // Get the maximum error between the two
    m_max_error(image_test, image_out, len0_src, len1_src);

    // Write output data to output file
    h_write_binary(image_out, "image_out.dat", nbytes_image);
    
    // Write kernel image to output file
    h_write_binary(image_kern, "image_kernel.dat", nbytes_kern);

    // Free memory
    free(image_in);
    free(image_out);
    free(image_test);
    
    // Reset compute devices
    h_reset_devices(num_devices);
}
