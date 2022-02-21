



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

#include <gnuradio/message_port_proxy.h>

namespace py = pybind11;

void bind_message_port_proxy(py::module& m) {

    py::class_<gr::message_port_proxy_upstream, gr::port_interface, std::shared_ptr<gr::message_port_proxy_upstream>>(
        m, "message_port_proxy_upstream")
        .def(py::init(&gr::message_port_proxy_upstream::make))
        .def("connect", &gr::message_port_proxy_upstream::connect)
        ;

    py::class_<gr::message_port_proxy_downstream, gr::port_interface, std::shared_ptr<gr::message_port_proxy_downstream>>(
        m, "message_port_proxy_downstream")
        .def(py::init(&gr::message_port_proxy_downstream::make))
        .def("port", &gr::message_port_proxy_downstream::port)
        .def("set_gr_port", &gr::message_port_proxy_downstream::set_gr_port)
        .def("start_rx", &gr::message_port_proxy_downstream::start_rx)
        ;
    
}

