
# Set all sources to be compiled with HIP compiler
set(sources
    ${CMAKE_CURRENT_SOURCE_DIR}/hip_helper.cpp
)

set(headers ${CMAKE_CURRENT_SOURCE_DIR}/hip_helper.hpp)

# Set the language of kernel sources
set_source_files_properties(${sources} PROPERTIES LANGUAGE ${kernel_lang})

# Add the hip_helper library as a static library
add_library (hip_helper 
    ${sources}
    ${headers}
)                

target_link_libraries(hip_helper ${kernel_libs})    

# Install the library as well
install(TARGETS hip_helper DESTINATION lib)
install(FILES ${headers} DESTINATION include)

target_include_directories(hip_helper PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

