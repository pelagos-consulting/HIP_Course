# Set all sources to be compiled with HIP compiler
set(headers ${CMAKE_CURRENT_SOURCE_DIR}/mat_size.hpp)

# Compile all targets in this directory using a loop
set(targets
    mat_mult_shared_A_col
    mat_mult_shared_A_flat_col
    mat_mult_tile_shared_AB_flat_col
    mat_mult_AT
    mat_mult_BT
    mat_mult_double
    mat_mult_float
    mat_mult_shared_A
    mat_mult_shared_B
    mat_mult_float_restrict
    mat_mult_tile_shared_A
    mat_mult_tile_shared_B
    mat_mult_tile_shared_AB
    mat_mult_tile_shared_A_vector
    mat_mult_tile_shared_B_vector
    mat_mult_tile_shared_AB_vector    
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

    # Link in other libraries
    target_link_libraries(${target}.exe hip_helper ${kernel_libs})                    
    # Set the linker language
    set_target_properties(${target}.exe
    	PROPERTIES LINKER_LANGUAGE ${linker_lang}
    )   
    # Install the target
    install(TARGETS ${target}.exe DESTINATION bin)

endforeach()

# Compile all targets in this directory using a loop
set(targets_hipblas
    mat_mult_float_hipblas
    mat_mult_float_md_hipblas
)

foreach(target ${targets_hipblas})
    # Set the language of kernel sources
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp 
        PROPERTIES LANGUAGE ${kernel_lang})
    
    # Add the hello_devices.exe executable for basic device discovery
    add_executable (${target}.exe
        ${CMAKE_CURRENT_SOURCE_DIR}/${target}.cpp
        ${headers}
    )

    # Add hipblas in the list of things to link to
    target_link_libraries(${target}.exe hip_helper ${kernel_libs} hipblas)
    
    # Set the linker language
    set_target_properties(${target}.exe
    	PROPERTIES LINKER_LANGUAGE ${linker_lang}
    )   
    # Install the target
    install(TARGETS ${target}.exe DESTINATION bin)

endforeach()







