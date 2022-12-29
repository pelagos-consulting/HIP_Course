        //// Begin Task 2 - Code to set kernel arguments for each thread /////////
        
        // Set kernel arguments for kernels[n]
        
        // Set kernel arguments here for convenience
        h_errchk(clSetKernelArg(kernels[n], 0, sizeof(buffer_srces[n]), &buffer_srces[n]), "Set kernel argument 0");
        h_errchk(clSetKernelArg(kernels[n], 1, sizeof(buffer_dests[n]), &buffer_dests[n]), "Set kernel argument 1");
        h_errchk(clSetKernelArg(kernels[n], 2, sizeof(buffer_kerns[n]), &buffer_kerns[n]), "Set kernel argument 2");
        h_errchk(clSetKernelArg(kernels[n], 3, sizeof(cl_int), &len0_src),  "Set kernel argument 3");
        h_errchk(clSetKernelArg(kernels[n], 4, sizeof(cl_int), &len1_src),  "Set kernel argument 4");
        h_errchk(clSetKernelArg(kernels[n], 5, sizeof(cl_int), &pad0_l),    "Set kernel argument 5");
        h_errchk(clSetKernelArg(kernels[n], 6, sizeof(cl_int), &pad0_r),    "Set kernel argument 6");
        h_errchk(clSetKernelArg(kernels[n], 7, sizeof(cl_int), &pad1_l),    "Set kernel argument 7");
        h_errchk(clSetKernelArg(kernels[n], 8, sizeof(cl_int), &pad1_r),    "Set kernel argument 8");
    
        //// End Task 2 //////////////////////////////////////////////////////////