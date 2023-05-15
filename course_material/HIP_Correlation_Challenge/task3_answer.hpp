            // Task 3 solution
            
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