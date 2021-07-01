/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "utils_pybind.hh"
#include <gnuradio/block_work_io.hh>
#include <gnuradio/buffer.hh>
#include <gnuradio/buffer_cpu_simple.hh>
#include <gnuradio/tag.hh>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
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
        .def("step", [](gr::block& gr_block, py::list arrays) {
            auto inputs = gr::list_to_work_inputs(arrays);
            auto outputs = gr::generate_block_work_outputs(gr_block, inputs[0].n_items);

            // We attempt to run the block. If the block cannot be run with a certain
            // size of buffers, we simply double the size. This is the part where we
            // are playing the role of the scheduler by having to
            // allocate/de-allocate stuff.
            unsigned int multiple = 1;
            auto work_status = gr_block.work(inputs, outputs);
            while (work_status ==
                   gr::work_return_code_t::WORK_INSUFFICIENT_OUTPUT_ITEMS) {
                work_status = gr_block.work(inputs, outputs);
                outputs = gr::generate_block_work_outputs(gr_block,
                                                          multiple * inputs[0].n_items);
                multiple *= 2;
            }

            // Finally, we want to convert everything back to numpy arrays here.

            return outputs;
        });
}
