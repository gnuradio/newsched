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

typedef py::array_t<int8_t, py::array::c_style | py::array::forcecast> numpy_byte_array_t;
typedef py::array_t<int16_t, py::array::c_style | py::array::forcecast>
    numpy_short_array_t;
typedef py::array_t<int32_t, py::array::c_style | py::array::forcecast> numpy_int_array_t;
typedef py::array_t<float, py::array::c_style | py::array::forcecast> numpy_float_array_t;
typedef py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>
    numpy_complex_float_array_t;

namespace gr {

template <typename T>
gr::block_work_input array_to_work_input(T input);

std::vector<gr::block_work_input> list_to_work_inputs(py::list arrays);

std::vector<block_work_output> generate_block_work_outputs(block gr_block,
                                                           size_t num_items);

std::vector<tag_t> dict_to_tags(py::dict dict);
py::dict tags_to_dict(std::vector<tag_t> tags);

template gr::block_work_input array_to_work_input(numpy_byte_array_t input);
template gr::block_work_input array_to_work_input(numpy_short_array_t input);
template gr::block_work_input array_to_work_input(numpy_int_array_t input);
template gr::block_work_input array_to_work_input(numpy_float_array_t input);
template gr::block_work_input array_to_work_input(numpy_complex_float_array_t input);

} // namespace gr