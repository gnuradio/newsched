
/*
 * flowgraphright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <thread>

namespace py = pybind11;

#include <gnuradio/flowgraph.h>
// pydoc.h is automatically generated in the build directory
// #include <block_pydoc.h>

void bind_flowgraph(py::module& m)
{
    py::class_<gr::flat_graph, gr::graph, gr::node, std::shared_ptr<gr::flat_graph>>(
        m, "flat_graph")
        ;

    py::class_<gr::flowgraph, gr::graph, gr::node, std::shared_ptr<gr::flowgraph>>(
        m, "flowgraph")
        .def(py::init(&gr::flowgraph::make), py::arg("name")="flowgraph")
        .def_static("check_connections", &gr::flowgraph::check_connections)
        .def("make_flat", &::gr::flowgraph::make_flat)

        .def("start", &::gr::flowgraph::start)
        .def("stop", &::gr::flowgraph::stop)
        .def("wait", &::gr::flowgraph::wait)
        .def("run", &::gr::flowgraph::run)
        ;
}
