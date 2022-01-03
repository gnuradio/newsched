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
        // .def_static(
        //     "make",
        //     py::overload_cast<gr::node_sptr, gr::port_sptr, gr::node_sptr,
        //     gr::port_sptr>(
        //         &::gr::edge::make))
        .def(py::init(
            [](gr::node_sptr a, gr::port_sptr b, gr::node_sptr c, gr::port_sptr d) {
                return ::gr::edge::make(a, b, c, d);
            }))
        .def("set_custom_buffer", &edge::set_custom_buffer)
        .def("identifier", &edge::identifier);
}
