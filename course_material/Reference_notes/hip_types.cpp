#include <cstdio>
#include <iostream>
#include <cstdint>
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
    
    // Example of complex numbers
    float2 num1 = make_float2(1.0f, 3.0f);
    float2 num2 = make_float2(4.0f, 6.0f);

    // Do complex arithmetic (1,3i)*(4,6i) = 4 - 18 + 12i + 6i = (-14, 18i)
    float2 num3 = (float2){0.0f};
    // Real part
    num3.x = num1.x*num2.x - num1.y*num2.y;
    // Complex part
    num3.y = num1.y*num2.x + num1.x*num2.y;
    // Print out the result
    std::cout << num3.x << " " << num3.y << "\n";

    // Using the dim3 type
    dim3 global_size = {512, 512, 1};
    dim3 block_size((uint32_t)16);

    // Print out the block size. Any unspecified dimensions will be filled with a value of 1
    std::cout << "Block size\n"; 
    std::cout << block_size.x << " " << block_size.y << " " << block_size.z << "\n";

    // Print out the global size
    std::cout << "Global size\n"; 
    std::cout << global_size.x << " " << global_size.y << " " << global_size.z << "\n"; 

}
