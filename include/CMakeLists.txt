# SPDX-FileCopyrightText: 2020 Marcus Müller <mueller@kit.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

install(FILES
  block.hpp
  block_work_io.hpp
  io_signature.hpp
  scheduler.hpp
  tag.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/newsched)
