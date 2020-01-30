# SPDX-FileCopyrightText: 2020 Marcus Müller <mueller@kit.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

add_library(newsched SHARED
  vector_source.cc
  vector_sink.cc
  multiply_const_blk.cc
  scheduler.cc
  
   
  )
set_target_properties(newsched PROPERTIES
            CXX_STANDARD 17
            )

#target_link_libraries(newsched PRIVATE spdlog::spdlog)
target_include_directories(newsched
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<INSTALL_INTERFACE:include>
  )
#pybind11_add_module(newschedpy pybind.cpp)
#target_link_libraries(newschedpy PUBLIC newsched)

#install(TARGETS newsched)
