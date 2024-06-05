// Main

#include <assert.h>
#include <cstdio>
#include <cstdint>
#include <omp.h>
#include <chrono>

#include "hip_helper.hpp"
#include "mat_helper.hpp"

#include "mat_size.hpp"

#define DEVEL

typedef float float_type;

__global__ void xcorr(
        float_type *src, 
        float_type *dst,
        float_type *kern,
        int len0_src,
        int len1_src, 
        int pad0_l,
        int pad0_r,
        int pad1_l,
        int pad1_r      
    ) {

    // get the coordinates
    size_t i0 = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i1 = blockIdx.x * blockDim.x + threadIdx.x;

    //// Task 1 complete the correlation kernel ////
    
    // Uncomment for the shortcut answer
    // #include "task1_answer.hpp" 

    // Reconstruct size of the kernel
    size_t len0_kern = pad0_l + pad0_r + 1;
    size_t len1_kern = pad1_l + pad1_r + 1;

    // Strides for the source and destination arrays
    size_t stride0_src = len1_src;
    size_t stride1_src = 1;

    // Strides for the cross-correlation kernel
    size_t stride0_kern = len1_kern;
    size_t stride1_kern = 1;

    // Assuming row-major ordering for arrays
    size_t offset_src = i0 * stride0_src + i1;
    size_t offset_kern = pad0_l*stride0_kern + pad1_l*stride1_kern; 

    if ((i0 >= pad0_l) && (i0 < len0_src-pad0_r) && (i1 >= pad1_l) && (i1 < len1_src-pad1_r)) {
        float_type sum = 0.0;
        for (int i = -pad0_l; i<= pad0_r; i++) {
            for (int j = -pad1_l; j <= pad1_r; j++) {
                sum += kern[offset_kern + i*stride0_kern + j*stride1_kern] 
                    * src[offset_src + i*stride0_src + j*stride1_src];
            }
        }
        dst[offset_src] = sum;
    }
    
    //// End Task 1 ////
}

int main(int argc, char** argv) {
    
    // Parse command line arguments
    int dev_index = h_parse_args(argc, argv);

    // Number of devices discovered
    int num_devices=0;

    // Helper function to acquire devices
    // This sets the default device
    h_acquire_devices(&num_devices, dev_index);
    
    // Report on each device
    for (int n=0; n<num_devices; n++) {
        h_report_on_device(n);
    }
    
    // Number of Bytes for a single image
    size_t nbytes_image = N0*N1*sizeof(float_type);

    // Number of Bytes for the stack of images
    size_t nbytes_input=NIMAGES*nbytes_image;
    // Output stack is the same size as the input
    size_t nbytes_output=nbytes_input;
    
    // Allocate storage for the output 
    float_type* images_out = (float_type*)h_alloc(
        nbytes_output, 
        (size_t)BYTE_ALIGNMENT
    );
    
    // Assume that images_in will have dimensions 
    // (NIMAGES, N0, N1) and will have row-major ordering
    size_t nbytes;
    
    // Read in the images
    float_type* images_in = (float_type*)h_read_binary("images_in.dat", &nbytes);
    assert(nbytes == nbytes_input);

    // Read in the image kernel
    size_t nbytes_image_kernel = (L0+R0+1)*(L1+R1+1)*sizeof(float_type);
    float_type* image_kernel = (float_type*)h_read_binary("image_kernel.dat", &nbytes);
    assert(nbytes == nbytes_image_kernel);

    // Create HIP buffers for source, destination, and image kernel
    float_type **srces_d = (float_type**)calloc(num_devices, sizeof(float_type*));
    float_type **dests_d = (float_type**)calloc(num_devices, sizeof(float_type*));
    float_type **kerns_d = (float_type**)calloc(num_devices, sizeof(float_type*));
   
    // Create input buffers for every device
    for (int n=0; n<num_devices; n++) {
        
        // Set the hip device to use
        H_ERRCHK(hipSetDevice(n));
        
        //// Begin Task 2 - Code to create the HIP buffers for each thread ////
        
        // Fill srces_d[n], dests_d[n], kerns_d[n] 
        // with buffers created by clCreateBuffer 
        // Use the h_errchk routine to check output
        
        // srces_d[n] is of size nbytes_image
        // dests_d[n] is of size nbytes_image
        // kerns_d[n] is of size nbytes_image_kernel
        
        // the array image_kernel contains the host-allocated 
        // memory for the image kernel
        
        // Uncomment for the shortcut answer
        // #include "task2_answer.hpp" 
        
        // Create buffers for sources
        H_ERRCHK(hipMalloc((void**)&srces_d[n], nbytes_image));
        H_ERRCHK(hipMalloc((void**)&dests_d[n], nbytes_image));
        H_ERRCHK(hipMalloc((void**)&kerns_d[n], nbytes_image_kernel));
        
        // Copy image kernel to device
        H_ERRCHK(
            hipMemcpy(
                kerns_d[n], 
                image_kernel,
                nbytes_image_kernel, 
                hipMemcpyHostToDevice)
        );

        // Set memory in dests_d using hipMemset
        H_ERRCHK(hipMemset(dests_d[n], 0, nbytes_image));
        
        //// End Task 2 //////////////////////////////////////////////////////////
    }
    
    // Keep track of how many images each device processed
    size_t* it_count = (size_t*)calloc(num_devices, sizeof(size_t)); 
    
    // Kernel size parameters
    dim3 block_size = { 8, 8, 1 };
    dim3 global_size = { (uint32_t)N1, (uint32_t)N0, 1 };
    dim3 grid_nblocks;

    // Choose the number of blocks so that grid fits within it.
    h_fit_blocks(&grid_nblocks, global_size, block_size);

#ifdef DEVEL

    
    

#else

    // Start the timer
    auto t1 = std::chrono::high_resolution_clock::now();  

    for (int i = 0; i<NITERS; i++) {
        printf("Processing iteration %d of %d\n", i+1, NITERS);
        
        #pragma omp parallel for default(none) schedule(dynamic, 1) num_threads(num_devices) \
            shared(block_size, global_size, grid_nblocks, images_in, images_out, \
                    dests_d, srces_d, kerns_d, nbytes_image, it_count)
        for (int n=0; n<NIMAGES; n++) {
            
            // Get the OpenMP thread ID
            int tid = omp_get_thread_num();
            
            // Set the compute device to use
            H_ERRCHK(hipSetDevice(tid));
            
            // Increment image counter for this device
            it_count[tid] += 1;
            
            // Load memory from images in using the offset
            size_t offset = n*N0*N1;
            
            //// Begin Task 3 - Code to upload memory to the compute device buffer ////
            
            // Uncomment for the shortcut answer
            // #include "task3_answer.hpp" 

            // Upload memory from images_in at offset
            // To srces_d[tid]
            H_ERRCHK(
                hipMemcpy(
                    srces_d[tid], 
                    &images_in[offset], 
                    nbytes_image, 
                    hipMemcpyHostToDevice
                )
            );

            //// End Task 3 ///////////////////////////////////////////////////////////
            
            
            //// Begin Task 4 - Code to launch the kernel ///////////////////////////
        
            // Uncomment for the shortcut answer
            // #include "task4_answer.hpp" 
        
            // Amount of shared memory to use in the kernel
            size_t sharedMemBytes=0;
            
            // Just for kernel arguments
            int len0_src = N0, len1_src = N1;
            int pad0_l = L0, pad0_r = R0, pad1_l = L1, pad1_r = R1;
            
            // Launch the kernel
            hipLaunchKernelGGL(xcorr, 
                grid_nblocks, 
                block_size, sharedMemBytes, 0, 
                srces_d[tid], dests_d[tid], kerns_d[tid],
                len0_src,
                len1_src, 
                pad0_l,
                pad0_r,
                pad1_l,
                pad1_r   
            );
            
            //// End Task 4 ///////////////////////////////////////////////////////////
            
            //// Begin Task 5 - Code to download memory from the compute device buffer
            
            // Uncomment for the shortcut answer
            // #include "task5_answer.hpp" 
            
            H_ERRCHK(
                hipMemcpy(
                    &images_out[offset], 
                    dests_d[tid], 
                    nbytes_image, 
                    hipMemcpyDeviceToHost
                )
            );
            
            //// End Task 5 ///////////////////////////////////////////////////////////
        }
    }

    // Stop the timer
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    double duration = time_span.count();
    
    // Get some statistics on how the computation is performing 
    int num_images = NITERS*NIMAGES;
    for (int i = 0; i< num_devices; i++) {
        //h_report_on_device(devices[i]);
        float_type pct = 100*(float_type)it_count[i]/(float_type)num_images;
        printf("Device %d processed %zu of %d images (%0.2f%%)\n", i, it_count[i], num_images, pct);
    }
    printf("Overall processing rate %0.2f images/s\n", (double)num_images/duration);

    // Write output data to output file
    h_write_binary(images_out, "images_out.dat", nbytes_output);
    
#endif
    
    // Free allocated memory
    free(image_kernel);
    free(images_in);
    free(images_out);
    free(it_count);

    // Release programs and kernels
    for (int n=0; n<num_devices; n++) {
        h_errchk(hipFree(srces_d[n]),"Releasing sources buffer");
        h_errchk(hipFree(dests_d[n]),"Releasing dests buffer");
        h_errchk(hipFree(kerns_d[n]),"Releasing image kernels buffer");
    }

    // Free memory
    free(srces_d);
    free(dests_d);
    free(kerns_d);
    
    // Reset compute devices
    h_reset_devices(num_devices);

}
