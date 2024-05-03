# Set all sources to be compiled with HIP compiler
set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/mat_mult.cpp
)
# Set the language of kernel sources
set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE ${kernel_lang})

# Add the hello_devices.exe executable for basic device discovery
add_executable (mat_mult.exe
    ${KERNEL_FILES}
)
# Link in other libraries
target_link_libraries(mat_mult.exe hip_helper ${kernel_libs})                    

set_target_properties(mat_mult.exe
	PROPERTIES LINKER_LANGUAGE ${linker_lang}
)   

# Install the target
install(TARGETS mat_mult.exe DESTINATION bin)





