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

#include <gnuradio/cudabuffer.hh>
// pydoc.h is automatically generated in the build directory
// #include <edge_pydoc.h>

void bind_cudabuffer(py::module& m)
{
    using cuda_buffer_properties = ::gr::cuda_buffer_properties;

    py::enum_<gr::cuda_buffer_type>(m, "cuda_buffer_type")
        .value("H2D", gr::cuda_buffer_type::H2D)
        .value("D2D", gr::cuda_buffer_type::D2D)
        .value("D2H", gr::cuda_buffer_type::D2H)
        .export_values();


    py::class_<cuda_buffer_properties,
               gr::buffer_properties,
               std::shared_ptr<cuda_buffer_properties>>(m, "cuda_buffer_properties")
        .def_static(
            "make", &cuda_buffer_properties::make, py::arg("buffer_type") = gr::cuda_buffer_type::D2D)
            ;
}
