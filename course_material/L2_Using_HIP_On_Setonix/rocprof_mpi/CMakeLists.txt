
set(HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_size.hpp
    ${PROJECT_SOURCE_DIR}/common/hip_helper.hpp
    ${PROJECT_SOURCE_DIR}/common/mat_helper.hpp
)

# Set all sources to be compiled with HIP compiler
set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_mult_profiling_mpi.cpp
    ${PROJECT_SOURCE_DIR}/common/hip_helper.cpp
)
# Set the language of kernel sources
set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE ${kernel_lang})

set(target mat_mult_profiling_mpi.exe)

# mat_mult_profiling_mpi.exe executable for profiling with rocprof and MPI
add_executable (${target}
    ${KERNEL_FILES}
    ${HEADER_FILES}
)
# Link in HIP and MPI
target_link_libraries(${target} ${kernel_libs} MPI::MPI_CXX)
# Include common directories
target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/common)
# Compile target with CXX compiler
set_target_properties(${target}
	PROPERTIES LINKER_LANGUAGE ${linker_lang}
)   
# Install the target
install(TARGETS ${target} DESTINATION bin)



