// Simple program to demonstrate a bug in hipMemcpy3D

#include <iostream>

// Include the size of arrays to be computed
#define N0_U 4
#define N1_U 4

// Bring in helper header to manage boilerplate code
#include "hip_helper.hpp"