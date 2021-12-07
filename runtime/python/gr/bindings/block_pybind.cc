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

#include <gnuradio/block.hh>
// pydoc.h is automatically generated in the build directory
// #include <block_pydoc.h>

void bind_block(py::module& m)
{
    using block = ::gr::block;

    py::class_<block, gr::node, std::shared_ptr<block>>(m, "block")
        .def("work",
             &block::work,
             py::arg("work_input_items"),
             py::arg("work_output_items"))
        .def("base",
            &block::base)
        .def("set_py_handle",
            &block::set_py_handle)
        .def("produce_each",
            &block::produce_each)
        .def("consume_each",
            &block::consume_each)
        ;

}
