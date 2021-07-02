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
gr::block_work_input array_to_input(T input)
{
    return gr::block_work_input(
        input.size(), input.itemsize(), reinterpret_cast<void*>(input.mutable_data()));
}

template gr::block_work_input array_to_input(numpy_byte_array_t input);
template gr::block_work_input array_to_input(numpy_short_array_t input);
template gr::block_work_input array_to_input(numpy_int_array_t input);
template gr::block_work_input array_to_input(numpy_float_array_t input);
template gr::block_work_input array_to_input(numpy_complex_float_array_t input);

std::vector<gr::block_work_input> list_to_inputs(py::list arrays)
{
    std::vector<gr::block_work_input> inputs;
    for (const auto& array : arrays) {
        if (py::isinstance<py::array_t<uint8_t>>(array)) {
            inputs.push_back(
                array_to_input<numpy_byte_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<int16_t>>(array)) {
            inputs.push_back(
                array_to_input<numpy_short_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<int32_t>>(array)) {
            inputs.push_back(
                array_to_input<numpy_int_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<float>>(array)) {
            inputs.push_back(
                array_to_input<numpy_float_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<std::complex<float>>>(array)) {
            inputs.push_back(array_to_input<numpy_complex_float_array_t>(
                pybind11::array::ensure(array)));
        } else {
            std::cerr << "Error in converting inputs to block_work_input" << std::endl;
        }
    }
    return inputs;
}


std::vector<block_work_output> generate_outputs(block& gr_block, size_t num_items)
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

std::vector<block_work_output> try_block_work(block& gr_block,
                                              std::vector<gr::block_work_input> inputs)
{
    unsigned int output_buffer_size = inputs[0].n_items;
    auto outputs = generate_outputs(gr_block, output_buffer_size);
    auto work_status = gr_block.work(inputs, outputs);

    // Double the output buffer size until it runs.
    while (work_status == gr::work_return_code_t::WORK_INSUFFICIENT_OUTPUT_ITEMS) {
        work_status = gr_block.work(inputs, outputs);
        outputs = generate_outputs(gr_block, output_buffer_size);
        output_buffer_size <<= 2;
    }

    return outputs;
}


py::list outputs_to_list(std::vector<block_work_output> outputs)
{
    py::list data;
    for (const auto& output : outputs) {
        auto array =
            py::array_t<float>(py::buffer_info(output.buffer->read_ptr(0),
                                               sizeof(float),
                                               py::format_descriptor<float>::format(),
                                               output.buffer->num_items(),
                                               true));
        data.append(array);
    }

    return data;
}


std::vector<std::vector<tag_t>> list_to_tags(py::list list_of_tags)
{
    std::vector<std::vector<tag_t>> tags;
    for (const py::dict& item : list_of_tags) {
        tags.push_back(dict_to_tag(item));
    }
    return tags;
}

std::vector<tag_t> dict_to_tag(py::dict dict)
{

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
}


py::dict tag_to_dict(std::vector<tag_t> tags)
{
    py::dict tag_dict;
    for (const auto& tag : tags) {
        // convert each item in the vector to an entry in the dict.
    }
}


py::list tags_to_list(std::vector<std::vector<tag_t>> tag_vector)
{
    py::list tag_list;
    for (const auto& tags : tag_vector)
        tag_list.append(tag_to_dict(tags));
    return tag_list;
}

} // namespace gr
