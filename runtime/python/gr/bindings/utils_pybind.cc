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
 * @brief This just wraps a buffer with a block_work_input. It's kind of awkward that
 * there's not complete type information being passed on from this function, just an
 * itemsize. Seems like it's asking for trouble later on.
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

std::vector<gr::block_work_input> list_to_inputs(block& gr_block, py::list arrays)
{
    // NOTE: This will take in a list of Python objects. We're hoping that these objects
    // are numpy arrays whose types match the input ports of the provided block. However,
    // we have to do some serious type checking, type casting, provide warnings when they
    // do not match or else users will get all kinds of wierd results when their blocks
    // process stuff

    // NOTE: Another note -- I just realized that blocks have message ports. How the heck
    // will I handle those from a "step" point of view. I was mostly thinking that any
    // time a developer is using "step", they just plain don't need messages and message
    // ports, but it would add that extra bit of continuity between initial protyping and
    // deployment. Actually, to do it would work because these blocks need to be connected
    // somehow and that's against how "step" would work.

    std::vector<gr::block_work_input> inputs;
    // This is assuming that input_ports returns the ports in the same order that their
    // respective inputs show up in arrays
    for (unsigned int i = 0; i < arrays.size(); i++) {
        auto array = arrays[i];
        auto port_data_type = gr_block.input_ports()[i]->data_type();

        bool is_byte = port_data_type == gr::param_type_t::UINT8 &&
                       py::isinstance<py::array_t<uint8_t>>(array);
        bool is_signed_char = port_data_type == gr::param_type_t::INT8 &&
                              py::isinstance<py::array_t<int8_t>>(array);
        bool is_short_int = port_data_type == gr::param_type_t::INT16 &&
                            py::isinstance<py::array_t<int16_t>>(array);
        bool is_int = port_data_type == gr::param_type_t::INT32 &&
                      py::isinstance<py::array_t<int32_t>>(array);
        bool is_float = port_data_type == gr::param_type_t::FLOAT &&
                        py::isinstance<py::array_t<float>>(array);
        bool is_complex_float = port_data_type == gr::param_type_t::CFLOAT &&
                                py::isinstance<py::array_t<gr_complex>>(array);

        // NOTE: So what I have here actually just checks the type and gives a warning.
        // What I want to do is actually cast the data type. This can also produce
        // unexpected results, but at least for some data types it would be just fine and
        // the results may be different, but not entirely unexpected.

        // NOTE: I've decided I don't really want to support casting. It's going to cause
        // less headaches for everyone if the user simply makes appropriate use of
        // .astype() function in numpy to get the inputs to be correct. As long as we
        // don't give back strange data-types, the user should only ever have to get the
        // first input block correct in terms of type. Even if the user changes the
        // data-type on their own with some intermediate numpy calculation, the numpy
        // .astype() is probably implemented very efficiently.
        if (!(is_byte || is_signed_char || is_short_int || is_short_int || is_int ||
              is_float || is_complex_float)) {
            std::cout << "Warning: provided data type of input array does not match "
                         "corresponding port data type. Interpreting bits as a different "
                         "numerical type can produce very unexpected results."
                      << std::endl;
        }

        if (py::isinstance<py::array_t<uint8_t>>(array)) {
            inputs.push_back(
                array_to_input<numpy_byte_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<int8_t>>(array)) {
            inputs.push_back(
                array_to_input<numpy_signed_char_array>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<int16_t>>(array)) {
            inputs.push_back(
                array_to_input<numpy_short_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<int32_t>>(array)) {
            inputs.push_back(
                array_to_input<numpy_int_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<float>>(array)) {
            inputs.push_back(
                array_to_input<numpy_float_array_t>(pybind11::array::ensure(array)));
        } else if (py::isinstance<py::array_t<gr_complex>>(array)) {
            inputs.push_back(array_to_input<numpy_complex_float_array_t>(
                pybind11::array::ensure(array)));
        } else {
            std::cout << "Unknown data type supplied for input " << i << std::endl;
        }
    }
    return inputs;
} // namespace gr


std::vector<block_work_output> generate_outputs(block& gr_block, size_t num_items)
{
    // TODO: This needs to allocate memory on the appropriate machine.
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


py::list outputs_to_list(block& gr_block, std::vector<block_work_output> outputs)
{
    py::list data;
    // This is assuming that output_ports returns the ports in the same order that
    // their respective outputs show up in outputs
    for (unsigned int i = 0; i < outputs.size(); i++) {
        // TODO: We're not considering some types here.
        auto port = gr_block.output_ports()[i];
        auto buf_ptr = outputs[i].buffer->read_ptr(0);
        auto num_items = outputs[i].buffer->num_items();
        auto item_size = outputs[i].buffer->item_size();
        switch (port->data_type()) {
        case gr::param_type_t::INT8:
            auto format = py::format_descriptor<uint8_t>::format();
            auto buf_info = py::buffer_info(buf_ptr, item_size, format, num_items, false);
            data.append(py::array_t<uint8_t>(buf_info));
            break;
        case gr::param_type_t::INT16:
            auto format = py::format_descriptor<int16_t>::format();
            auto buf_info = py::buffer_info(buf_ptr, item_size, format, num_items, false);
            data.append(py::array_t<int16_t>(buf_info));
            break;
        case gr::param_type_t::INT32:
            auto format = py::format_descriptor<int32_t>::format();
            auto buf_info = py::buffer_info(buf_ptr, item_size, format, num_items, false);
            data.append(py::array_t<int32_t>(buf_info));
            break;
        case gr::param_type_t::FLOAT:
            auto format = py::format_descriptor<float>::format();
            auto buf_info = py::buffer_info(buf_ptr, item_size, format, num_items, false);
            data.append(py::array_t<float>(buf_info));
            break;
        case gr::param_type_t::CFLOAT:
            auto format = py::format_descriptor<float>::format();
            auto buf_info = py::buffer_info(buf_ptr, item_size, format, num_items, false);
            data.append(py::array_t<gr_complex>(buf_info));
            break;
        default: // All other types are essentially treated as "bytes"
            auto format = py::format_descriptor<uint8_t>::format();
            auto buf_info = py::buffer_info(buf_ptr, item_size, format, num_items, false);
            data.append(py::array_t<uint8_t>(buf_info));
            break;
        };
    }

    return data;
}


std::vector<std::vector<tag_t>> list_to_tags(py::list list_of_tags)
{
    std::vector<std::vector<tag_t>> tags;
    for (const auto& item : list_of_tags) {
        auto dict = pybind11::dict(item, true);
        if (!dict.check()) {
            std::cerr << "Tags should be a Python dict. Unable to convert provided "
                         "Python object to tag. Ignoring: "
                      << item << std::endl;
            continue;
        }
        tags.push_back(dict_to_tag(dict));
    }
    return tags;
}

std::vector<tag_t> dict_to_tag(py::dict dict)
{
    auto offset = dict("offset");
    auto key = dict("key");
    auto value = dict("value");
    auto src_id = dict("src_id");

    // key_pmt = pmtf::PmtBuilder().add_data()

    // if (pybind11::str(key, true).check()) {
    // } else if (pybind11::array(key, true).check()) {
    // } else if (pybind11::int_(key, true).check()) {
    // } else if (pybind11::bool_(key, true).check()) {
    // } else if (pybind11::float_(key, true).check()) {
    // }

    // return tag_t(offset, key_pmt, value_pmt, srcid_pmt);
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
