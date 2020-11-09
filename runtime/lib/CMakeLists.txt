add_library(gnuradio-runtime SHARED
  # parameter.cpp
  parameter_types.cpp
  block.cpp
  sync_block.cpp
  domain_adapter_zmq.cpp
  flat_graph.cpp
  flowgraph_monitor.cpp
  flowgraph.cpp
  logging.cpp
  scheduler.cpp
  realtime.cpp
  )
set_target_properties(gnuradio-runtime PROPERTIES
            CXX_STANDARD 17
            )

target_include_directories(gnuradio-runtime
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${YAML_CPP_INCLUDE_DIR}
  )

target_link_libraries(gnuradio-runtime
PUBLIC
Volk::volk
spdlog::spdlog
ZeroMQ::ZeroMQ
${YAML_CPP_LIBRARIES}

)

# ZMQ 4.3.1 deprecated the send and recv with references.  Ignore for now // FIXME
target_compile_options(gnuradio-runtime PRIVATE -Wno-deprecated-declarations)

if(GR_ENABLE_CUDA)
add_library(gnuradio-runtime-cu SHARED
  cudabuffer.cu
  cudabuffer_pinned.cu
  )
  target_include_directories(gnuradio-runtime-cu
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include 
  )

  target_link_libraries(gnuradio-runtime
  PUBLIC
  gnuradio-runtime-cu)

  set_target_properties(gnuradio-runtime-cu PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(gnuradio-runtime-cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()
