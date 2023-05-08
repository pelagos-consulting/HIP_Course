// Include the HIP helper headers
#include <hip/hip_runtime.h>

/// Examine an error code and exit if necessary.
void h_errchk(hipError_t errcode, const char* message) {

    if (errcode != hipSuccess) { 
        const char* errstring = hipGetErrorString(errcode); 
        std::fprintf( 
            stderr, 
            "Error, HIP call failed at %s, error string is: %s\n", 
            message, 
            errstring 
        ); 
        exit(EXIT_FAILURE); 
    }
}

/// Macro to check error codes.
#define H_ERRCHK(cmd) \
{\
    h_errchk(cmd, "__FILE__:__LINE__");\
}

// Main program
int main(int argc, char** argv) {
    
    // Initialise HIP explicitly
    H_ERRCHK(hipInit(0));
    
    // Make sure dev_index is sane
    int num_devices=0;
    H_ERRCHK(hipGetDeviceCount(&num_devices));
    
    // Print some properties of each compute device
    for (int i = 0; i<num_devices; i++) {
        // Set the compute device to use, 
        // this also makes sure a primary context is available
        H_ERRCHK(hipSetDevice(i));
        
        // Report some information on a compute device
        hipDeviceProp_t prop;

        // Get the properties of the compute device
        H_ERRCHK(hipGetDeviceProperties(&prop, i));

        // ID of the compute device
        std::printf("Device id: %d\n", i);

        // Name of the compute device
        std::printf("\t%-19s %s\n","name:", prop.name);

        // Size of global memory
        std::printf("\t%-19s %lu MB\n","global memory size:",prop.totalGlobalMem/(1000000)); 
    }
    
    // Reset devices to clean up resources
    for (int i = 0; i<num_devices; i++) {
        // Set device
        H_ERRCHK(hipSetDevice(i));

        // Synchronize device 
        H_ERRCHK(hipDeviceSynchronize());

        // Reset device (destroys primary context)
        H_ERRCHK(hipDeviceReset());
    }
}
