# SPDX-FileCopyrightText: 2020 Marcus Müller <mueller@kit.edu>
#

cmake_minimum_required(VERSION 3.10)
project(newsched)

#list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_STANDARD 14)
# narf – C++17 + Clang + Pybind11 requires workaround: https://github.com/pybind/pybind11/issues/1818
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsized-deallocation")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The version number.
set(newsched_VERSION_MAJOR 0)
set(newsched_VERSION_MINOR 0)
set(newsched_VERSION_PATCH 1)
set(VERSION "${newsched_VERSION_MAJOR}.${newsched_VERSION_MINOR}.${newsched_VERSION_PATCH}")
set(LIBVER "${newsched_VERSION_MAJOR}.${newsched_VERSION_MINOR}")

#find_package(Python3)
#find_package(spdlog REQUIRED)

#enable_testing()
#add_subdirectory(pybind11)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
