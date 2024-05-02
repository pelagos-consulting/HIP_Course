            // Task 5 solution

            H_ERRCHK(
                hipMemcpy(
                    &images_out[offset], 
                    dests_d[tid], 
                    nbytes_image, 
                    hipMemcpyDeviceToHost
                )
            );