add_library(pmt SHARED
  pmt.cpp
  )

message(STATUS ${PROJECT_SOURCE_DIR})

target_include_directories(pmt
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/pmt
  )