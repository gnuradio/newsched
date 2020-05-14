########################################################################
# Install header files
########################################################################
install(FILES
  block.hpp
  block_callbacks.hpp
  io_signature.hpp
  sync_block.hpp
  tag.hpp
  types.hpp
  DESTINATION ${GR_INCLUDE_DIR}/gnuradio/blocklib/
)

message(STATUS ${CMAKE_INSTALL_INCLUDEDIR})