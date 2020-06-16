add_library(gnuradio-runtime SHARED
  flat_graph.cpp   
  scheduler.cpp
  )
set_target_properties(gnuradio-runtime PROPERTIES
            CXX_STANDARD 17
            )

target_include_directories(gnuradio-runtime
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  ${CMAKE_SOURCE_DIR}/blocklib/common/include
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

target_link_libraries(gnuradio-runtime 
Volk::volk
gnuradio-blocklib-blocks)

message(STATUS ${INTERFACE_INCLUDE_DIRECTORIES})
#pybind11_add_module(newschedpy pybind.cpp)
#target_link_libraries(newschedpy PUBLIC newsched)

#install(TARGETS newsched)