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

#include <gnuradio/vmcircbuf.hh>
// pydoc.h is automatically generated in the build directory
// #include <edge_pydoc.h>

void bind_vmcircbuf(py::module& m)
{
    using vmcirc_buffer_properties = ::gr::vmcirc_buffer_properties;

    py::enum_<gr::vmcirc_buffer_type>(m, "vmcirc_buffer_type")
        .value("AUTO", gr::vmcirc_buffer_type::AUTO)
        .value("SYSV_SHM", gr::vmcirc_buffer_type::SYSV_SHM)
        .value("MMAP_SHM", gr::vmcirc_buffer_type::MMAP_SHM)
        .value("MMAP_TMPFILE", gr::vmcirc_buffer_type::MMAP_TMPFILE)
        .export_values();


    py::class_<vmcirc_buffer_properties,
               gr::buffer_properties,
               std::shared_ptr<vmcirc_buffer_properties>>(m, "vmcirc_buffer_properties")
        .def_static(
            "make", &vmcirc_buffer_properties::make, py::arg("buffer_type") = gr::vmcirc_buffer_type::AUTO)
            ;
}
