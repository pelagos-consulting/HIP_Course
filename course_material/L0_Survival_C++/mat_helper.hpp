
// Define the size of the matrix


// Function to pretty print a matrix
void m_show_matrix(float* src, size_t N0, size_t N1) {

    for (size_t i1=0; i1<N1; i1++) {
        std::printf("-");
    }

    std::printf("\n");

    // Pretty print the matrix
    for (size_t i0=0; i0<N0; i0++) {

        for (size_t i1=0; i1<N1; i1++) {
            size_t offset = i0*N1 + i1;

            if (i1==0) {
                std::printf("|");
            }

            std::printf(" %9.2e", src[offset]);


            if (i1 == N1-1) {
                std::printf(" |\n");
            }
        }
    }

    for (size_t i1=0; i1<N1; i1++) {
        std::printf("-");
    }

    std::printf("\n");
}

