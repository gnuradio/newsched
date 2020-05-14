add_library(gnuradio-scheduler-simplepacket SHARED
  scheduler_simplepacket.cpp   
  )
set_target_properties(gnuradio-runtime PROPERTIES
            CXX_STANDARD 14
            )

message(STATUS ${PROJECT_SOURCE_DIR})

target_include_directories(gnuradio-scheduler-simplestream
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/gnuradio/schedulers/simplepacket
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/include 
  )

  # TODO - learn how to include against runtime with INSTALL_INTERFACE

  target_link_libraries(gnuradio-scheduler-simplepacket PUBLIC
    gnuradio-runtime
    Boost::boost
    pthread)