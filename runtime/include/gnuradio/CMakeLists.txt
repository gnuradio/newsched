########################################################################
# Install header files
########################################################################
install(FILES
  api.h
  attributes.h
  flowgraph.hpp
  scheduler.hpp
  DESTINATION ${GR_INCLUDE_DIR}/gnuradio
)

if(GR_ENABLE_CUDA)
install(FILES
  cudabuffer.hpp
  DESTINATION ${GR_INCLUDE_DIR}/gnuradio
)
endif()

message(STATUS ${CMAKE_INSTALL_INCLUDEDIR})