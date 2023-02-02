    // Initialise HIP explicitly
    H_ERRCHK(hipInit(0));

    // Get the number of devices
    H_ERRCHK(hipGetDeviceCount(&num_devices));

    // Check to make sure we have one or more suitable devices,
    // and we chose a sane index
    if ((num_devices == 0) || (dev_index >= num_devices)) {
        std::printf("Failed to find a suitable compute device\n");
        exit(EXIT_FAILURE);
    }

    // Set the device to use
    H_ERRCHK(hipSetDevice(dev_index));
