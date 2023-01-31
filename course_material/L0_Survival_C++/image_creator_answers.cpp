#include <cstdio>
#include <iostream>

// Helper routines to work with matrices
#include "mat_helper.hpp"

// Get the size of the matrices
#include "mat_size.hpp"

// Function to compute the square of two numbers
float square(int x, int y) {
    return (float)x * (float)y;
}

// Main program
int main(int argc, char** argv) {
    
    { // Starting a code block
        // Declaring integers
        char a_i=1;         // Using char as an integer
        short b_i=4;        // 16 bit
        int c_i=2;          // 32 bit          
        unsigned int d_i=3; // 32 bit
        long e_i = 5;       // 64 bit
    }

    // This is a comment
    char a='s'; // Using char as a character
    char b = a+1; // Arithmetic with characters
    std::printf("b interpreted as an integer: %i\n", b); // print b with the memory interpreted as an integer
    std::printf("b interpreted as a character: %c\n", b); // print b with the memory interpreted as a character

    // Making a string from characters
    char str[] = {'a', 'b', 'c', 'd', '\0'};
    std::printf("%s\n", str);

    // Declaring floating point value
    float x=5.0;
    double y=6.0;
    long double z=7.0;

    // Printing floats
    std::printf("float representation, x = %f y = %f\n", x, y); // Print x and y to the screen with their memory interpreted as floats

    // The constants N0 and N1 were defined in the file mat_size.hpp

    // Make up strides for multi-dimensional indexing, use row-major ordering
    int s0 = N1;
    int s1 = 1;

    // Make up array and fill it using nested for loops
    float *arr = (float*)calloc((size_t)(N0*N1), sizeof(float));

    for (int i0=0; i0 < N0; i0++) {
        for (int i1=0; i1 < N1; i1++) {
            // Use the dot product to make up the position in the allocation
            int offset = i0*s0 + i1*s1;
    
            // Fill the allocation by calling a function
            arr[offset] = square(i0, i1);
        }
    }

    // Print the array
    m_show_matrix(arr, (size_t)N0, (size_t)N1);

    // Write the array to disk
    const char *fname = "image.dat";
    FILE *fp = fopen(fname, "wb");

    // Sanity check using an "if" statement
    if (fp != NULL) {
        fwrite(arr, sizeof(float), (size_t)(N0*N1), fp);
    } else {
        std::printf("File %s could not be opened\n", fname);
    }

    // Close the file and free the array
    fclose(fp);
    free(arr);
}
