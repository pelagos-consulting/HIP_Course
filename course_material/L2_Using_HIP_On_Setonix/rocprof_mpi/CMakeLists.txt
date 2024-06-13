# Set the header files
set(HEADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_size.hpp
    ${PROJECT_SOURCE_DIR}/common/hip_helper.hpp
    ${PROJECT_SOURCE_DIR}/common/mat_helper.hpp
)

# Set all sources to be compiled as host+device code
set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_mult_profiling_mpi.cpp
    ${PROJECT_SOURCE_DIR}/common/hip_helper.cpp
)
# Set the language of kernel sources
set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE ${kernel_lang})

set(target mat_mult_profiling_mpi.exe)

# Add the executable for compilation
add_executable (${target}
    ${KERNEL_FILES}
    ${HEADER_FILES}
)
# Link in kernel_libs and MPI and any extra MPI libs
target_link_libraries(${target} ${kernel_libs} MPI::MPI_CXX $ENV{MPI_EXTRA_LIBS})
# Include common directories
target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/common)
# Compile target with CXX compiler
set_target_properties(${target}
	PROPERTIES LINKER_LANGUAGE ${linker_lang}
)   
# Install the target
install(TARGETS ${target} DESTINATION bin)



