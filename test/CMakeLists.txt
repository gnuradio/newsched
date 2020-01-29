# SPDX-FileCopyrightText: 2020 Marcus Müller <mueller@kit.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

include(CTest)

#include(FindPy)
#PYTHON_TRY_MODULE("PyTest" "pytest" HAVE_PYTEST)
#if(NOT HAVE_PYTEST)
#  message(FATAL_ERROR "Can't test without pytest. Please install.")
#endif()

# add_test(NAME instantiation
#   COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}
#   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../)
