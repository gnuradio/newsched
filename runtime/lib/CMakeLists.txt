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

message(STATUS ${INTERFACE_INCLUDE_DIRECTORIES})
#pybind11_add_module(newschedpy pybind.cpp)
#target_link_libraries(newschedpy PUBLIC newsched)

#install(TARGETS newsched)
