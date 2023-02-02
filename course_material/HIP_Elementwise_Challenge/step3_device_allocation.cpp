    // Allocate memory on device for arrays D, E, and F
    float *D_d, *E_d, *F_d;
    H_ERRCHK(hipMalloc((void**)&D_d, nbytes_D));
    H_ERRCHK(hipMalloc((void**)&E_d, nbytes_E));
    H_ERRCHK(hipMalloc((void**)&F_d, nbytes_F));

