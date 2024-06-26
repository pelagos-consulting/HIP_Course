// Main

#include <assert.h>
#include <cstdio>
#include <cstdint>
#include <omp.h>
#include <chrono>

#include "hip_helper.hpp"
#include "mat_size.hpp"

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
        nbytes_output
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




            //// End Task 3 ///////////////////////////////////////////////////////////
            
            
            //// Begin Task 4 - Code to launch the kernel ///////////////////////////
        
            // Uncomment for the shortcut answer
            // #include "task4_answer.hpp" 
        
        
        
            //// End Task 4 ///////////////////////////////////////////////////////////
            
            //// Begin Task 5 - Code to download memory from the compute device buffer
            
            // Uncomment for the shortcut answer
            // #include "task5_answer.hpp" 
            

            
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
