/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/port.hh>
// pydoc.h is automatically generated in the build directory
// #include <port_pydoc.h>

void bind_port(py::module& m)
{
    using port = ::gr::port_base;

    py::class_<port, std::shared_ptr<::gr::port_base>>(m, "port")
    .def("set_custom_buffer", &port::set_custom_buffer)
        ;

}
