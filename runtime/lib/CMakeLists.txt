add_library(gnuradio-runtime SHARED
  block.cpp
  flowgraph.cpp   
  scheduler.cpp
  sync_block.cpp
  )
set_target_properties(gnuradio-runtime PROPERTIES
            CXX_STANDARD 14
            )

target_include_directories(gnuradio-runtime
  PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

message(STATUS ${INTERFACE_INCLUDE_DIRECTORIES})
#pybind11_add_module(newschedpy pybind.cpp)
#target_link_libraries(newschedpy PUBLIC newsched)

#install(TARGETS newsched)