/* -*- c++ -*- */
/*
 * Copyright 2004,2008,2010,2013,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "vector_sink.hpp"
#include <algorithm>
#include <iostream>

using namespace std;
namespace gr {
namespace blocks {

template <class T>
vector_sink<T>::vector_sink(unsigned int vlen, const int reserve_items)
    : sync_block("vector_sink"),
      d_vlen(vlen)
{
    add_port(port<T>::make("input",
                               port_direction_t::INPUT,
                               port_type_t::STREAM,
                               std::vector<size_t>{ vlen }));

    // add_param(param<unsigned int>(vector_sink_params::k, "k", 1.0));

    
    std::scoped_lock guard(d_data_mutex);
    d_data.reserve(d_vlen * reserve_items);
}

template <class T>
const std::vector<T> vector_sink<T>::data()
{
    std::scoped_lock guard(d_data_mutex);
    return d_data;
}

template <class T>
const std::vector<tag_t> vector_sink<T>::tags()
{
    std::scoped_lock guard(d_data_mutex);
    return d_tags;
}


template <class T>
void vector_sink<T>::reset()
{
    std::scoped_lock guard(d_data_mutex);
    d_tags.clear();
    d_data.clear();
}

template <class T>
work_return_code_t vector_sink<T>::work(std::vector<block_work_input>& work_input,
                                          std::vector<block_work_output>& work_output)
{
    T* iptr = (T*)work_input[0].items;
    int noutput_items = work_input[0].n_items;

    // can't touch this (as long as work() is working, the accessors shall not
    // read the data
    std::scoped_lock guard(d_data_mutex);
    for (unsigned int i = 0; i < noutput_items * d_vlen; i++)
        d_data.push_back(iptr[i]);
    // std::vector<tag_t> tags;
    // this->get_tags_in_range(
    //     tags, 0, this->nitems_read(0), this->nitems_read(0) + noutput_items);
    // d_tags.insert(d_tags.end(), tags.begin(), tags.end());
    d_tags.insert(d_tags.end(), work_input[0].tags.begin(), work_input[0].tags.end());

    work_input[0].n_consumed = noutput_items;
    return work_return_code_t::WORK_OK;
}

template class vector_sink<std::uint8_t>;
template class vector_sink<std::int16_t>;
template class vector_sink<std::int32_t>;
template class vector_sink<float>;
template class vector_sink<gr_complex>;
} /* namespace blocks */
} /* namespace gr */