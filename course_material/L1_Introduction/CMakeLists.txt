
# Set all sources to be compiled with HIP compiler
set(KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/hello_devices.cpp
)
# Set the language of kernel sources
set_source_files_properties(${KERNEL_FILES} PROPERTIES LANGUAGE ${kernel_lang})

# Add the hello_devices.exe executable for basic device discovery
add_executable (hello_devices.exe
    ${KERNEL_FILES}
)
# Link in other libraries
target_link_libraries(hello_devices.exe ${kernel_libs})                    

set_target_properties(hello_devices.exe
	PROPERTIES LINKER_LANGUAGE CXX
)   


# Install the target
install(TARGETS hello_devices.exe DESTINATION bin)





