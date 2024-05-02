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
# CXX compiler flags
if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "-fPIC -fopenmp -pthread")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
    set(CMAKE_CXX_FLAGS_PROFILE "-pg -O3")
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgomp -lpthread")
endif()

if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-fPIC -fopenmp -pthread")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_COVERAGE "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
    set(CMAKE_CXX_FLAGS_PROFILE "-pg -O3")
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lomp -lpthread")
endif()

# ------ dependencies section ------ #

# HIP
find_package(hip REQUIRED)

# Set language for kernel code
set(kernel_lang HIP)

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
    ## This is to make sure nvcc is compiling for the same architecture
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=$ENV{GPU_ARCH} -Xcompiler -fopenmp")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
    set(CMAKE_CUDA_FLAGS_PROFILE "-pg -O3")
    set(CMAKE_CUDA_FLAGS_RELEASE "-g -O3")

    ## Change global preprocessor defintions for CUDA sources
        
    ## Remove any preprocessor definitions for AMD
    remove_definitions(-D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__)
    ## Replace it with CUDA precprocessor definitions
    add_definitions(-D__HIP_PLATFORM_NVCC__ -D__HIP_PLATFORM_NVIDIA__)  
else()
    message("Building for an AMD backend.")
    
    # Set flags for the compiler that will compile HIP sources
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} --offload-arch=$ENV{GPU_ARCH} -fopenmp")
    set(CMAKE_HIP_FLAGS_DEBUG "-g -ggdb -O1")
    set(CMAKE_HIP_FLAGS_PROFILE "-pg -O3")
    set(CMAKE_HIP_FLAGS_RELEASE "-g -O3")
endif()

# ----- end dependencies section ----- #

# Example Codes
add_subdirectory(${CMAKE_SOURCE_DIR}/L1_Introduction)
