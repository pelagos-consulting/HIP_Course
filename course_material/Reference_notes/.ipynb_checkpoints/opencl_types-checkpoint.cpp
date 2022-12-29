#include <cstdio>
#include <iostream>

#include "cl_helper.hpp"

// Main program
int main(int argc, char** argv) {

    // Declare an initialised vector
    cl_float4 f = (cl_float4){0.0, 1.0, 2.0, 3.0};
    
    // Could have also been done like this
    //cl_float4 f = (cl_float4){0.0};
    
    // Print out the last element
    std::printf("%f\n", f.s[3]);
    
    // Store a value in the last element
    f.s[3] = 10.0;
    
    // Print out the last element again
    std::printf("%f\n", f.s[3]);    
}
