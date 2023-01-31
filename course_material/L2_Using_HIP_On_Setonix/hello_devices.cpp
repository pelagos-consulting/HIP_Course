// Include the OpenCL helper headers
#include "hip_helper.hpp"

// Main program
int main(int argc, char** argv) {
    // Parse command line arguments
    int dev_index = h_parse_args(argc, argv);  

    // The number of devices
    int num_devices=0;

    // Discover devices and set resources
    h_acquire_devices(&num_devices, dev_index);

    // Report on available devices
    for (int n=0; n<num_devices; n++) {
        std::cout << "device " << n << std::endl;
        h_report_on_device(n);
    }
    
    // Release devices and contexts
    h_reset_devices(num_devices);
}
