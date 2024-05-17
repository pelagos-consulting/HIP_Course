        // Task 2 solution

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