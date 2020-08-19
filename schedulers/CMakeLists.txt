add_subdirectory(simplestream)

if(GR_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
