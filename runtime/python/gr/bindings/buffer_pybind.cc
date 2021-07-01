/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/buffer.hh>
// pydoc.h is automatically generated in the build directory
// #include <edge_pydoc.h>

void bind_buffer(py::module& m)
{
    using buffer_properties = ::gr::buffer_properties;
    using buffer_reader = ::gr::buffer_reader;
    using buffer = ::gr::buffer;

    py::class_<buffer_properties, std::shared_ptr<buffer_properties>>(
        m, "buffer_properties");

    py::class_<buffer_reader, std::shared_ptr<buffer_reader>>(m, "buffer_reader")
        .def("numpy", [](buffer_reader& reader) {
            return py::array_t<float>(
                py::buffer_info(reader.read_ptr(),
                                sizeof(float),
                                py::format_descriptor<float>::format(),
                                reader.num_items(),
                                true));
        });

    py::class_<buffer, std::shared_ptr<buffer>>(m, "buffer")
        .def("numpy", [](buffer& buf) {
            return py::array_t<float>(
                py::buffer_info(buf.read_ptr(0),
                                sizeof(float),
                                py::format_descriptor<float>::format(),
                                buf.num_items(),
                                true));
        });
}
