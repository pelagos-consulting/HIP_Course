    H_ERRCHK(hipMemcpy(D_d, D_h, nbytes_D, hipMemcpyHostToDevice));
    H_ERRCHK(hipMemcpy(E_d, E_h, nbytes_E, hipMemcpyHostToDevice));
