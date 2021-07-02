/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once
#include "utils_pybind.hh"
#include <gnuradio/block_work_io.hh>
#include <gnuradio/buffer.hh>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/block.hh>


// typedefs for handling multiple types of arrays
typedef py::array_t<int8_t, py::array::c_style | py::array::forcecast> numpy_byte_array_t;
typedef py::array_t<int16_t, py::array::c_style | py::array::forcecast>
    numpy_short_array_t;
typedef py::array_t<int32_t, py::array::c_style | py::array::forcecast> numpy_int_array_t;
typedef py::array_t<float, py::array::c_style | py::array::forcecast> numpy_float_array_t;
typedef py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>
    numpy_complex_float_array_t;

namespace gr {

/**
 * @brief
 *
 * @tparam T
 * @param input
 * @return gr::block_work_input
 */
template <typename T>
gr::block_work_input array_to_input(T input);

/**
 * @brief
 *
 * @param arrays
 * @return std::vector<gr::block_work_input>
 */
std::vector<gr::block_work_input> list_to_inputs(py::list arrays);

/**
 * @brief
 *
 * @param gr_block
 * @param num_items
 * @return std::vector<block_work_output>
 */
std::vector<block_work_output> generate_outputs(block& gr_block, size_t num_items);


/**
 * @brief
 *
 * @param gr_block
 * @param inputs
 * @return std::vector<block_work_output>
 */
std::vector<block_work_output> try_block_work(block& gr_block,
                                              std::vector<gr::block_work_input> inputs);
/**
 * @brief
 *
 * @param outputs
 * @return py::list
 */
py::list outputs_to_list(std::vector<block_work_output> outputs);


/**
 * @brief
 *
 * @param list_of_tags
 * @return std::vector<tag_t>
 */
std::vector<std::vector<tag_t>> list_to_tags(py::list list_of_tags);

/**
 * @brief
 *
 * @param dict
 * @return std::vector<tag_t>
 */
std::vector<tag_t> dict_to_tag(py::dict dict);

/**
 * @brief
 *
 * @param tags
 * @return py::dict
 */
py::dict tag_to_dict(std::vector<tag_t> tags);

/**
 * @brief
 *
 * @param tag_vector
 * @return py::list
 */
py::list tags_to_list(std::vector<std::vector<tag_t>> tag_vector);

} // namespace gr