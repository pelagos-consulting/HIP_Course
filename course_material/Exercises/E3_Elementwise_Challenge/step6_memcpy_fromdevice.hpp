    // Use hipMemcpy to copy memory from F_d to F_h
    H_ERRCHK(hipMemcpy(F_h, F_d, nbytes_F, hipMemcpyDeviceToHost));
