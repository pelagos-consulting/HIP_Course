# Set all sources to be compiled with HIP compiler
set(headers ${CMAKE_CURRENT_SOURCE_DIR}/mat_size.hpp)

# Compile all targets in this directory using a loop
set(targets 
    mat_mult_bug
    mat_mult_memcheck
    mat_mult_printf
)

foreach(target ${targets})
    # Set the language of kernel sources
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp 
        PROPERTIES LANGUAGE ${kernel_lang})
    
    # Add the hello_devices.exe executable for basic device discovery
    add_executable (${target}.exe
        ${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp
        ${headers}
    )

    # Set debugging flags for CUDA or AMD
    if (CUDAToolkit_FOUND)
        # Add debugging flags for CUDA
        target_compile_options(${target}.exe PRIVATE -g -O0)
    else()
        # Add debugging flags for AMD
        target_compile_options(${target}.exe PRIVATE -g -ggdb -O1)
    endif()

    # Link in other libraries
    target_link_libraries(${target}.exe hip_helper ${kernel_libs})                    
    # Set the linker language
    set_target_properties(${target}.exe
    	PROPERTIES LINKER_LANGUAGE ${linker_lang}
    )   
    # Install the target
    install(TARGETS ${target}.exe DESTINATION bin)

endforeach()











