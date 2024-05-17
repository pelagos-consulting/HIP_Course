
# Set all sources to be compiled with HIP compiler
set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_devices_mpi_onefile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_devices_mpi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_jobstep.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels.hip.cpp
)
# Set the language of kernel sources
set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE ${kernel_lang})

# hello_devices_mpi_onefile.exe executable for basic device discovery - one file
add_executable (hello_devices_mpi_onefile.exe
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_devices_mpi_onefile.cpp
)
# Link in HIP and MPI
target_link_libraries(hello_devices_mpi_onefile.exe ${kernel_libs} MPI::MPI_CXX $ENV{MPI_EXTRA_LIBS})                    
set_target_properties(hello_devices_mpi_onefile.exe
	PROPERTIES LINKER_LANGUAGE ${linker_lang}
)   
# Install the target
install(TARGETS hello_devices_mpi_onefile.exe DESTINATION bin)


# hello_devices_mpi.exe executable for basic device discovery - kernel is separate
add_executable (hello_devices_mpi.exe
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_devices_mpi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels.hip.cpp
)
# Link in HIP and MPI
target_link_libraries(hello_devices_mpi.exe ${kernel_libs} MPI::MPI_CXX $ENV{MPI_EXTRA_LIBS})                    
# Compile target with CXX compiler
set_target_properties(hello_devices_mpi.exe
	PROPERTIES LINKER_LANGUAGE ${linker_lang}
)   
# Install the target
install(TARGETS hello_devices_mpi.exe DESTINATION bin)


# hello_jobstep.exe for discovery of MPI process, 
add_executable (hello_jobstep.exe
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_jobstep.cpp
)
# Link in HIP and MPI
target_link_libraries(hello_jobstep.exe ${kernel_libs} MPI::MPI_CXX $ENV{MPI_EXTRA_LIBS})                    
# Compile target with CXX compiler
set_target_properties(hello_jobstep.exe
	PROPERTIES LINKER_LANGUAGE ${linker_lang}
)   
# Install the target
install(TARGETS hello_jobstep.exe DESTINATION bin)
