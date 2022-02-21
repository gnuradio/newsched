



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
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#include <gnuradio/runtime.h>

namespace py = pybind11;

void bind_runtime(py::module& m) {

    py::class_<gr::runtime, std::shared_ptr<gr::runtime>>(
        m, "runtime")

        .def(py::init(&gr::runtime::make))
        .def("add_scheduler",  py::overload_cast<std::pair<gr::scheduler_sptr, std::vector<gr::node_sptr>>>(&gr::runtime::add_scheduler), py::arg("conf"))
        .def("add_scheduler",  py::overload_cast<gr::scheduler_sptr>(&gr::runtime::add_scheduler), py::arg("sched"))
        .def("add_proxy", &gr::runtime::add_proxy)
        // .def("clear_schedulers", &gr::flowgraph::clear_schedulers)
        .def("initialize", &gr::runtime::initialize)
        .def("start", &gr::runtime::start)
        .def("stop", &gr::runtime::stop)
        .def("wait", &gr::runtime::wait, py::call_guard<py::gil_scoped_release>())
        .def("run", &gr::runtime::run, py::call_guard<py::gil_scoped_release>());
    
}

