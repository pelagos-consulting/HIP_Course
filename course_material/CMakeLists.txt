cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21 or greater
cmake_policy(VERSION 3.21...3.27)

if(NOT DEFINED ROCM_PATH)
    if(NOT DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "/opt/rocm/" CACHE PATH "Path to which ROCm has been installed")
    else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
    endif()
endif()

project(hip_course VERSION 1.0.0
    DESCRIPTION "Example codes to accompany a course in HIP"
    LANGUAGES HIP C CXX
)

message(${CMAKE_CXX_COMPILER_ID})

# Default CXX compiler flags for different compilers
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

# ------ dependencies section ------ #

find_package(MPI REQUIRED)

# HIP
find_package(hip REQUIRED)

# Set language for kernel code
set(kernel_lang HIP)

# Set this variable as a placeholder language for linking most C++ sources
set(linker_lang CXX)

# Set libraries for kernel code
set(kernel_libs hip::device)

if ((DEFINED ENV{HIP_PLATFORM}) AND ($ENV{HIP_PLATFORM} STREQUAL nvidia))
    message("Building for an NVIDIA backend.")
        
    # CUDA language must be enabled first
    enable_language(CUDA)
    set(CUDA_SEPARABLE_COMPILATION ON)
        
    ## Find the CUDA toolkit, 
    ## it must be present if we use HIP_PLATFORM=nvidia 
    find_package(CUDAToolkit REQUIRED)
        
    set(kernel_lang CUDA)
        
    ## Append the CUDA toolkit libraries to kernel_libs
    ##list(APPEND kernel_libs CUDA::cudart CUDA::cuda_driver)

    ## Point kernel_libs to the CUDA toolkit libraries
    set(kernel_libs CUDA::cudart CUDA::cuda_driver)
        
    ## Set your CUDA GPU architectures to use here
    set(CMAKE_CUDA_ARCHITECTURES $ENV{GPU_ARCH})

    ## CUDA compile flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -Xcompiler -fopenmp -Xcompiler -fPIC")
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -O0")
    set(CMAKE_CUDA_FLAGS_PROFILE "-pg -O3")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

    ## Change global preprocessor defintions for CUDA sources
        
    ## Remove any preprocessor definitions for AMD
    remove_definitions(-D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__)
    ## Replace it with CUDA precprocessor definitions
    add_definitions(-D__HIP_PLATFORM_NVCC__ -D__HIP_PLATFORM_NVIDIA__)  
else()
    message("Building for an AMD backend.")
    
    ## Set your HIP GPU architectures to use here
    set(CMAKE_HIP_ARCHITECTURES $ENV{GPU_ARCH})

    ## HIP compile flags
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g -fopenmp")
    set(CMAKE_HIP_FLAGS_DEBUG "-ggdb -O1")
    set(CMAKE_HIP_FLAGS_PROFILE "-pg -O3")
    set(CMAKE_HIP_FLAGS_RELEASE "-O3")
endif()

# ----- end dependencies section ----- #

# Set common directory a the destination for all includes
include_directories(common)

# Make sure the all target doesn't get brought in for install
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
##set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
##set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)

# Example Codes
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
