add_library(gnuradio-scheduler-cuda-cu SHARED
  cudabuffer.cu
  )


  target_include_directories(gnuradio-scheduler-cuda-cu
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/gnuradio/schedulers/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/include 
  )


add_library(gnuradio-scheduler-cuda SHARED
  scheduler_cuda.cpp   
  )
set_target_properties(gnuradio-runtime PROPERTIES
            CXX_STANDARD 17
            )

message(STATUS ${PROJECT_SOURCE_DIR})

target_include_directories(gnuradio-scheduler-cuda
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/gnuradio/schedulers/cuda
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/include 
  )

  # TODO - learn how to include against runtime with INSTALL_INTERFACE

  target_link_libraries(gnuradio-scheduler-cuda PUBLIC
    gnuradio-runtime
    Boost::boost
    pthread
    gnuradio-scheduler-cuda-cu)

# libzmq3-dev on Ubuntu 20.04 throws out deprecation warnings, but rolling back fails on CI
# FIXME: just hide the deprecation warnings for now
target_compile_options(gnuradio-scheduler-cuda PRIVATE -Wno-deprecated-declarations)

set_target_properties(gnuradio-blocklib-cuda-cu PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(gnuradio-blocklib-cuda-cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)