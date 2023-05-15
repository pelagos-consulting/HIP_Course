            // Task 4 solution

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