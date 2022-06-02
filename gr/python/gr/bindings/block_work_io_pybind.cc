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

#include <gnuradio/block_work_io.h>
// pydoc.h is automatically generated in the build directory
// #include <block_pydoc.h>

void bind_block_work_io(py::module& m)
{
    py::enum_<gr::work_return_code_t>(m, "work_return_t")
        .value("WORK_ERROR", gr::work_return_code_t::WORK_ERROR) // -100
        .value("WORK_INSUFFICIENT_OUTPUT_ITEMS",
               gr::work_return_code_t::WORK_INSUFFICIENT_OUTPUT_ITEMS) // -3
        .value("WORK_INSUFFICIENT_INPUT_ITEMS",
               gr::work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS) // -2
        .value("WORK_DONE", gr::work_return_code_t::WORK_DONE)        // -1
        .value("WORK_OK", gr::work_return_code_t::WORK_OK)            //  0
        .export_values();

    py::class_<gr::block_work_output, std::shared_ptr<gr::block_work_output>>(
        m, "block_work_output")
        .def_readwrite("n_items", &gr::block_work_output::n_items)
        .def_readwrite("buffer", &gr::block_work_output::buffer)
        .def_readwrite("n_produced", &gr::block_work_output::n_produced)

        .def("raw_items", &gr::block_work_output::raw_items)
        .def("produce", &gr::block_work_output::produce);

    py::class_<gr::block_work_input, std::shared_ptr<gr::block_work_input>>(
        m, "block_work_input")
        .def_readwrite("n_items", &gr::block_work_input::n_items)
        .def_readwrite("buffer", &gr::block_work_input::buffer)
        .def_readwrite("n_consumed", &gr::block_work_input::n_consumed)

        .def("nitems_read", &gr::block_work_input::nitems_read)
        .def("raw_items", &gr::block_work_input::raw_items)
        .def("consume", &gr::block_work_input::consume)
        .def("tags_in_window", &gr::block_work_input::tags_in_window);
}
