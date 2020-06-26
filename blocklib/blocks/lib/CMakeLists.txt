########################################################################
# Create a New CMake Target for this Library
########################################################################
add_library (
  gnuradio-blocklib-blocks
  SHARED
  multiply_const_blk.cpp
  vector_sink.cpp
  vector_source.cpp
)

########################################################################
# Specify directories to include for this target
########################################################################
target_include_directories (
  gnuradio-blocklib-blocks
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include/gnuradio/blocklib/blocks
          ${PROJECT_SOURCE_DIR}/blocklib/common/include
)

# TODO - learn how to include against runtime with INSTALL_INTERFACE
########################################################################
# Specify libraries to link against for this target
########################################################################
target_link_libraries (
  gnuradio-blocklib-blocks
  PUBLIC gnuradio-blocklib-common
         Volk::volk
)

########################################################################
# Specify compile options for this target
########################################################################
# Compile options specified in the root CMakeLists.txt can simply be propagated here.
# target_compile_options (gnuradio-blocklib-blocks)
