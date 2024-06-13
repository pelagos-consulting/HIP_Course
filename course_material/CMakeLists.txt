cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21 or greater
cmake_policy(VERSION 3.21...3.27)

# Make sure ROCM_PATH is defined
if(NOT DEFINED ROCM_PATH)
    if(NOT DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "/opt/rocm/" CACHE PATH "Path to which ROCm has been installed")
    else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
    endif()
endif()

# Set up the project
project(hip_course VERSION 1.0.0
    DESCRIPTION "Example codes to accompany a course in HIP"
    LANGUAGES HIP C CXX
)

message("Compiler ID is: ${CMAKE_CXX_COMPILER_ID}")

# Default compiler flags for the C++ compiler

if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-g -fPIC -fopenmp -pthread")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
    set(CMAKE_CXX_FLAGS_PROFILE "-pg -O3")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -lpthread")
endif()

if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-g -fPIC -fopenmp -pthread")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
    set(CMAKE_CXX_FLAGS_PROFILE "-pg -O3")
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -lpthread")
endif()

# Find packages
find_package(MPI REQUIRED)
find_package(hip REQUIRED)

# Query HIP_PLATFORM and setup kernel_lang as HIP or CUDA
if ((DEFINED ENV{HIP_PLATFORM}) AND ($ENV{HIP_PLATFORM} STREQUAL nvidia))
    message("Building for an NVIDIA backend.")
        
    # CUDA language must be enabled first
    enable_language(CUDA)
    set(CUDA_SEPARABLE_COMPILATION ON)
        
    # Find the CUDA toolkit, 
    # it is required if the environment variable HIP_PLATFORM is "nvidia" 
    find_package(CUDAToolkit REQUIRED)

    # Set the language for all host+device code
    set(kernel_lang CUDA)

    # Set kernel_libs to the CUDA toolkit libraries
    set(kernel_libs CUDA::cudart CUDA::cuda_driver)
        
    # Set the CUDA GPU architectures to use
    set(CMAKE_CUDA_ARCHITECTURES $ENV{GPU_ARCH})

    # Compile flags for different build types
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -Xcompiler -fopenmp -Xcompiler -fPIC")
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0")
    set(CMAKE_CUDA_FLAGS_PROFILE "-pg -O3")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

    # Change global preprocessor definitions for CUDA sources.
    # These become switches for AMD/NVIDIA in hip/hip_runtime.h
        
    # Remove AMD preprocessor defines
    remove_definitions(-D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__)
    # Set CUDA preprocessor defines
    add_definitions(-D__HIP_PLATFORM_NVCC__ -D__HIP_PLATFORM_NVIDIA__)  

else()
    message("Building for an AMD backend.")

    # Set the language for device code
    set(kernel_lang HIP)

    # Set libraries for kernel code
    set(kernel_libs hip::device)
    
    # Set your HIP GPU architectures to use here
    set(CMAKE_HIP_ARCHITECTURES $ENV{GPU_ARCH})

    # HIP compile flags
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g -fPIC -fopenmp")
    set(CMAKE_HIP_FLAGS_DEBUG "-ggdb -O1")
    set(CMAKE_HIP_FLAGS_PROFILE "-pg -O3")
    set(CMAKE_HIP_FLAGS_RELEASE "-O3")
endif()

# ----- end dependencies section ----- #

# Set linker_lang as the language used to link all sources
set(linker_lang CXX)

# Set common directory as the destination for all includes in builds
include_directories(common)

# Make sure the all target doesn't get brought in for every install
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
##set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
##set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)

# Add all directories that participate in the codebase.
add_subdirectory(${CMAKE_SOURCE_DIR}/common)
add_subdirectory(${CMAKE_SOURCE_DIR}/L1_Introduction)
add_subdirectory(${CMAKE_SOURCE_DIR}/L2_Using_HIP_On_Setonix)
add_subdirectory(${CMAKE_SOURCE_DIR}/L2_Using_HIP_On_Setonix/rocprof_mpi)
add_subdirectory(${CMAKE_SOURCE_DIR}/L3_Matrix_Multiplication)
add_subdirectory(${CMAKE_SOURCE_DIR}/L4_Debugging)
add_subdirectory(${CMAKE_SOURCE_DIR}/L5_Profiling)
add_subdirectory(${CMAKE_SOURCE_DIR}/L6_Memory_Management)
add_subdirectory(${CMAKE_SOURCE_DIR}/L7_Kernel_Optimisation)
add_subdirectory(${CMAKE_SOURCE_DIR}/L8_IO_Optimisation)
add_subdirectory(${CMAKE_SOURCE_DIR}/Exercises/E3_Elementwise_Challenge)
add_subdirectory(${CMAKE_SOURCE_DIR}/Exercises/E4_Debugging)
add_subdirectory(${CMAKE_SOURCE_DIR}/Exercises/E5_Profiling)
add_subdirectory(${CMAKE_SOURCE_DIR}/Exercises/E6_Memory)
add_subdirectory(${CMAKE_SOURCE_DIR}/Exercises/E8_Concurrent_IO)
add_subdirectory(${CMAKE_SOURCE_DIR}/Exercises/E10_Correlation_Challenge)
