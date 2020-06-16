add_library(gnuradio-blocklib-blocks SHARED
  multiply_const_blk.cpp
  vector_sink.cpp
  vector_source.cpp
  )


target_include_directories(gnuradio-blocklib-blocks
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/gnuradio/blocklib/blocks
  ${PROJECT_SOURCE_DIR}/blocklib/common/include 
  )

  # TODO - learn how to include against runtime with INSTALL_INTERFACE

  target_link_libraries(gnuradio-blocklib-blocks PUBLIC
  gnuradio-blocklib-common Volk::volk)