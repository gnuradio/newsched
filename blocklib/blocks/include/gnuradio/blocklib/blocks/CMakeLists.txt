########################################################################
# Create install rule for header files
########################################################################
install (
  FILES multiply_const_blk.hpp
        vector_source.hpp
        vector_sink.hpp
  DESTINATION ${GR_INCLUDE_DIR}/gnuradio/blocklib/blocks/
)
