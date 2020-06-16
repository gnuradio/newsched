add_library(gnuradio-blocklib-common SHARED
  parameter_types.cpp
  block.cpp
  sync_block.cpp
  
  )
set_target_properties(gnuradio-blocklib-common PROPERTIES
            CXX_STANDARD 17
            )

target_include_directories(gnuradio-blocklib-common
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