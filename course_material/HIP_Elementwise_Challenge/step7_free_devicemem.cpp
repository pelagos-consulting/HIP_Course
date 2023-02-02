    // Free the HIP buffers
    H_ERRCHK(hipFree(D_d));
    H_ERRCHK(hipFree(E_d));
    H_ERRCHK(hipFree(F_d));
