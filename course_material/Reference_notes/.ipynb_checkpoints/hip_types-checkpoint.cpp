#include <cstdio>
#include <iostream>

#include "hip_helper.hpp"

// Main program
int main(int argc, char** argv) {

    // Declare an initialised vector
    float4 f = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    
    // Could have also been initialised like this
    float4 v = (float4){0.0};
    
    // Print out the last element of v
    std::printf("%f\n", v.w);
    
    // Store a value in the last element
    f.w = 10.0f;
    
    // Print out the last element of v
    std::printf("%f\n", f.w);
    
    // Allocate a datatype of type long1
    long1 large = (long1){0};
    std::printf("long1 has %zu bytes \n", sizeof(large));
    
    // Allocate a datatype of type longlong1
    longlong1 huge = (longlong1){0};
    std::printf("longlong1 has %zu bytes \n", sizeof(huge));
    
    
    
}
