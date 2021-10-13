
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

#include <gnuradio/flowgraph.hh>
// pydoc.h is automatically generated in the build directory
// #include <block_pydoc.h>

void bind_flowgraph(py::module& m)
{
    py::class_<gr::flowgraph, gr::graph, gr::node, std::shared_ptr<gr::flowgraph>>(
        m, "flowgraph")

        .def(py::init(&gr::flowgraph::make))
        .def("set_scheduler", &gr::flowgraph::set_scheduler, py::arg("sched"))
        .def("set_schedulers", &gr::flowgraph::set_schedulers, py::arg("sched"))
        .def("add_scheduler", &gr::flowgraph::add_scheduler, py::arg("sched"))
        .def("clear_schedulers", &gr::flowgraph::clear_schedulers)
        .def("partition", &gr::flowgraph::partition)
        .def("validate", &gr::flowgraph::validate)
        .def("start", &gr::flowgraph::start)
        .def("stop", &gr::flowgraph::stop)
        .def("wait",
             [](gr::flowgraph& self) {
                 std::thread th([] {
                     for (;;) {
                         if (PyErr_CheckSignals() != 0)
                             throw py::error_already_set();
                         std::this_thread::sleep_for(
                             std::chrono::milliseconds(100));
                     }
                 });
                 th.detach();
                 self.wait();
             })
        .def("run", [](gr::flowgraph& self) {
            std::thread th([] {
                for (;;) {
                    if (PyErr_CheckSignals() != 0)
                        throw py::error_already_set();
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(100));
                }
            });
            th.detach();
            self.run();
        });
}
