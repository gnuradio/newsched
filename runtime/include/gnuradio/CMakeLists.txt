########################################################################
# Install header files
########################################################################
install(FILES
  api.h
  attributes.h
  block.hpp
  block_callbacks.hpp
  flowgraph.hpp
  io_signature.hpp
  scheduler.hpp
  sync_block.hpp
  tag.hpp
  types.hpp
  DESTINATION ${GR_INCLUDE_DIR}/gnuradio
)

message(STATUS ${CMAKE_INSTALL_INCLUDEDIR})