add_library(gnuradio-blocklib-cuda-cu SHARED
  multiply_const.cu
  multiply_const2.cu
  )

  add_library(gnuradio-blocklib-cuda SHARED
  multiply_const.cpp
  multiply_const2.cpp
  )

target_include_directories(gnuradio-blocklib-cuda
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/gnuradio/blocklib/cuda
  PRIVATE
  ${PROJECT_SOURCE_DIR}/blocklib/common/include
  ${PROJECT_SOURCE_DIR}/runtime/include
  )

  # TODO - learn how to include against runtime with INSTALL_INTERFACE

  target_link_libraries(gnuradio-blocklib-cuda PUBLIC
  gnuradio-runtime gnuradio-blocklib-cuda-cu )

  set_target_properties(gnuradio-blocklib-cuda-cu PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(gnuradio-blocklib-cuda-cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)