/* Code to perform a Matrix multiplication using OpenCL
Written by Dr Toby M. Potter
*/

//// Step 1. Setup headers and parse command line arguments ////

#include <cassert>
#include <cmath>
#include <iostream>

// Bring in helper header to manage boilerplate code
#include "hip_helper.hpp"

int main(int argc, char** argv) {

    // Get the size of differnt data types
    std::printf("Size of char1: %zu\n", sizeof(char1));
    std::printf("Size of short1: %zu\n", sizeof(short1));
    std::printf("Size of int1: %zu\n", sizeof(int1));
    std::printf("Size of long1: %zu\n", sizeof(long1));    
    std::printf("Size of longlong1: %zu\n", sizeof(longlong1));
    std::printf("Size of float1: %zu\n", sizeof(float1));    
    std::printf("Size of double1: %zu\n", sizeof(double1));

    // One way to make a vector type
    float4 temp = (float4){0.0f, 1.0f, 2.0f, 3.0f};
    
    // Access to individual elements in a vector
    temp.x = 0.0f;
    temp.y = 1.0f;
    temp.z = 2.0f;
    temp.w = 3.0f;
    
    // Shortened quick form to make vector
    float4 temp2 = (float4){0.0f};
    
    // Another way to make a vector type using the make_<vector type> function
    float4 temp3 = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
    
    std::printf("temp.w is %f\n", temp.w);
    
}

