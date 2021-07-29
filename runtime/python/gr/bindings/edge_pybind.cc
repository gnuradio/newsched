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

#include <gnuradio/edge.hh>
// pydoc.h is automatically generated in the build directory
// #include <edge_pydoc.h>

void bind_edge(py::module& m)
{
    using edge = ::gr::edge;

    py::class_<edge, std::shared_ptr<edge>>(m, "edge")
    .def("set_custom_buffer", &edge::set_custom_buffer)
        ;

}
