/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "utils_pybind.hh"


namespace gr {
/**
 * @brief
 *
 * @param input
 * @return std::vector<gr::block_work_input>
 */
template <typename T>
gr::block_work_input array_to_work_input(T input)
{
    return gr::block_work_input::make(
        input.size(), input.itemsize(), reinterpret_cast<void*>(input.mutable_data()));
}

std::vector<gr::block_work_input> list_to_work_inputs(py::list arrays)
{
    std::vector<gr::block_work_input> inputs;
    for (const auto& array : arrays) {
        if (py::isinstance<py::array_t<uint8_t>>(array)) {
            inputs.push_back(
                array_to_work_input<numpy_byte_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<int16_t>>(array)) {
            inputs.push_back(
                array_to_work_input<numpy_short_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<int32_t>>(array)) {
            inputs.push_back(
                array_to_work_input<numpy_int_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<float>>(array)) {
            inputs.push_back(
                array_to_work_input<numpy_float_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<std::complex<float>>>(array)) {
            inputs.push_back(array_to_work_input<numpy_complex_float_array_t>(
                pybind11::array::ensure(array)));
        } else {
            std::cerr << "Error in converting inputs to block_work_input" << std::endl;
        }
    }
    return inputs;
}


std::vector<block_work_output> generate_block_work_outputs(block gr_block,
                                                           size_t num_items)
{
    std::vector<block_work_output> outputs;
    for (unsigned int i = 0; i < gr_block.output_ports().size(); i++) {
        auto item_size = gr_block.output_ports()[i]->itemsize();
        auto outbuf_props = std::make_shared<buffer_properties>();
        outbuf_props->set_buffer_size(num_items * item_size);
        auto buffer =
            std::make_shared<buffer_cpu_simple>(num_items, item_size, outbuf_props);
        outputs.push_back(block_work_output(num_items, buffer));
    }

    return outputs;
}

std::vector<tag_t> dict_to_tags(py::dict dict){

    // Essentially I need to convert from py::dict to PMTs
    // TODO [GV]: I think converting tags can be done later because it's not
    // clear how we'll do tags at this point anyway.

    //  for (auto item : tags) {
    //      auto key = item.first;
    //      auto tag = tag_t(0, pmtf::PmtBuilder());
    //      if (pybind11::str(key, true).check()) {
    //      } else if (pybind11::array(key, true).check()) {
    //      } else if (pybind11::int_(key, true).check()) {
    //      } else if (pybind11::bool_(key, true).check()) {
    //      } else if (pybind11::float_(key, true).check()) {
    //      }
    //  }
};

py::dict tags_to_dict(std::vector<tag_t> tags){};

} // namespace gr
