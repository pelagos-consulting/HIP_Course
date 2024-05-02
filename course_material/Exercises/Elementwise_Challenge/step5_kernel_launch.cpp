    // Run the kernel
    hipLaunchKernelGGL(mat_elementwise, 
            grid_nblocks, 
            block_size, 0, 0, 
            D_d, E_d, F_d,
            N0_F,
            N1_F
    );

    // Wait for any commands to complete on the compute device
    H_ERRCHK(hipDeviceSynchronize());
