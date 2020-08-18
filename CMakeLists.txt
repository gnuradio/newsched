# SPDX-FileCopyrightText: 2020 Marcus Müller <mueller@kit.edu>
#

cmake_minimum_required(VERSION 3.10)
project(modulargr)

message(STATUS ${CMAKE_BUILD_TYPE})

SET(CMAKE_CXX_FLAGS_NOOPTWITHASM "-Wall -save-temps -g -O0" CACHE STRING
"Flags used by the C++ compiler during NoOptWithASM builds." FORCE)
SET(CMAKE_C_FLAGS_NOOPTWITHASM "-Wall -save-temps -g -O0" CACHE STRING
"Flags used by the C compiler during NoOptWithASM builds." FORCE)
SET(CMAKE_EXE_LINKER_FLAGS_NOOPTWITHASM
"-Wl,--warn-unresolved-symbols,--warn-once" CACHE STRING
"Flags used for linking binaries during NoOptWithASM builds." FORCE)
SET(CMAKE_SHARED_LINKER_FLAGS_NOOPTWITHASM
"-Wl,--warn-unresolved-symbols,--warn-once" CACHE STRING
"Flags used by the shared lib linker during NoOptWithASM builds." FORCE)

# Make sure our local CMake Modules path comes first
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake/Modules)

MARK_AS_ADVANCED(
CMAKE_CXX_FLAGS_NOOPTWITHASM
CMAKE_C_FLAGS_NOOPTWITHASM
CMAKE_EXE_LINKER_FLAGS_NOOPTWITHASM
CMAKE_SHARED_LINKER_FLAGS_NOOPTWITHASM)

set(CMAKE_CXX_STANDARD 17)
# narf – C++17 + Clang + Pybind11 requires workaround: https://github.com/pybind/pybind11/issues/1818
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsized-deallocation")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

########################################################################
# Incorporate CMake function/macros overloading.
########################################################################

include(CMakeOverloads)

########################################################################
# Compiler specific setup
########################################################################
include(GrMiscUtils) #compiler flag check

# The version number.
set(modulargr_VERSION_MAJOR 0)
set(modulargr_VERSION_MINOR 0)
set(modulargr_VERSION_PATCH 1)
set(VERSION "${modulargr_VERSION_MAJOR}.${modulargr_VERSION_MINOR}.${modulargr_VERSION_PATCH}")
set(LIBVER "${modulargr_VERSION_MAJOR}.${modulargr_VERSION_MINOR}")

SET(GR_INCLUDE_DIR     include CACHE PATH "Path to install header files")

set(BOOST_REQUIRED_COMPONENTS
program_options
)
set(GR_BOOST_MIN_VERSION "1.65")
find_package(Boost ${GR_BOOST_MIN_VERSION} COMPONENTS ${BOOST_REQUIRED_COMPONENTS})

find_package(Volk REQUIRED)

#find_package(Python3)
find_package(ZeroMQ REQUIRED)

enable_testing()

add_subdirectory(deps/catch/include)
add_subdirectory(blocklib)
add_subdirectory(runtime)
add_subdirectory(schedulers)
add_subdirectory(bench)
add_subdirectory(docs)

