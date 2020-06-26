########################################################################
# Project Meta-data
########################################################################
cmake_minimum_required (VERSION 3.10)
project (modulargr)

########################################################################
# Project Version Information
########################################################################
set (modulargr_VERSION_MAJOR 0)
set (modulargr_VERSION_MINOR 0)
set (modulargr_VERSION_PATCH 1)

set (VERSION "${modulargr_VERSION_MAJOR}.${modulargr_VERSION_MINOR}.${modulargr_VERSION_PATCH}")
set (LIBVER "${modulargr_VERSION_MAJOR}.${modulargr_VERSION_MINOR}")

########################################################################
# Dependency Versioning
########################################################################
set (VOLK_MIN_VERSION "2.1.0")
set (BOOST_REQUIRED_COMPONENTS)
set (GR_BOOST_MIN_VERSION "1.53")

########################################################################
# Build Options
########################################################################
option (ENABLE_TESTS "Enable tests" OFF)
option (ENABLE_PYBIND11 "Generate PyBIND11 bindings" OFF)

########################################################################
# Project Configuration
########################################################################
# Make sure our local CMake Modules path comes first
list (
  INSERT
  CMAKE_MODULE_PATH
  0
  ${CMAKE_SOURCE_DIR}/cmake/Modules
)

# Set compiler flags
# [GV] I'm thinking target_compile_options() or something similar would be more appropriate
set (
  CMAKE_CXX_FLAGS_NOOPTWITHASM
  "-Wall -save-temps -g -O0"
  CACHE STRING "Flags used by the C++ compiler during NoOptWithASM builds." FORCE
)
set (
  CMAKE_C_FLAGS_NOOPTWITHASM
  "-Wall -save-temps -g -O0"
  CACHE STRING "Flags used by the C compiler during NoOptWithASM builds." FORCE
)
set (
  CMAKE_EXE_LINKER_FLAGS_NOOPTWITHASM
  "-Wl,--warn-unresolved-symbols,--warn-once"
  CACHE STRING "Flags used for linking binaries during NoOptWithASM builds." FORCE
)
set (
  CMAKE_SHARED_LINKER_FLAGS_NOOPTWITHASM
  "-Wl,--warn-unresolved-symbols,--warn-once"
  CACHE STRING "Flags used by the shared lib linker during NoOptWithASM builds." FORCE
)

mark_as_advanced (
  CMAKE_CXX_FLAGS_NOOPTWITHASM
  CMAKE_C_FLAGS_NOOPTWITHASM
  CMAKE_EXE_LINKER_FLAGS_NOOPTWITHASM
  CMAKE_SHARED_LINKER_FLAGS_NOOPTWITHASM
)

# Force c++17? I'm realy not sure this part's necessary
set (CMAKE_CXX_STANDARD 17)
# narf – C++17 + Clang + Pybind11 requires workaround: https://github.com/pybind/pybind11/issues/1818
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsized-deallocation")
endif ()
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

set (
  GR_INCLUDE_DIR
  include
  CACHE PATH "Path to install header files"
)

########################################################################
# Find Dependencies
########################################################################

find_package (Boost ${GR_BOOST_MIN_VERSION} COMPONENTS ${BOOST_REQUIRED_COMPONENTS})

find_package (Volk REQUIRED)
# find_package (Python3)
# find_package (spdlog REQUIRED)

########################################################################
# Process other directories
########################################################################
# enable_testing()
# add_subdirectory(pybind11)
add_subdirectory (blocklib)
add_subdirectory (runtime)
# add_subdirectory (plugin)
add_subdirectory (schedulers)
add_subdirectory (docs)
