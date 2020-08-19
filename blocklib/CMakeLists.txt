add_subdirectory(blocks)

if(GR_ENABLE_CUDA)
add_subdirectory(cuda)
endif()
