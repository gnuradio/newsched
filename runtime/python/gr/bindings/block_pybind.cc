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
        .def("step", [](gr::block& gr_block, py::list arrays, py::list tags) {
            auto inputs = gr::list_to_inputs(arrays);
            auto pmt_tags = gr::list_to_tags(tags);
            // pmt_tags need to be added onto the inputs vector

            auto outputs = gr::try_block_work(gr_block, inputs);

            // At the moment, we cannot access the tags that have been appended/added to
            // the gr_block_outputs to be returned back to Python, so I suppose inputs and
            // outputs both need that functionality. I don't see why it would be harmful
            // for inputs/outputs to both have the ability to get/set all tag stuff.

            // std::vector<std::vector<tag_t>> output_tags;
            // for (unsigned int i = 0; i < gr_block.output_ports().size(); i++) {
            //     output_tags.push_back(outputs[i]);
            // }
            auto output_tags =
                gr::tag_to_dict(inputs[0].tags_in_window(0, inputs[0].nitems_read()));
            return py::make_tuple(outputs_to_list(outputs), output_tags);
        });
}
