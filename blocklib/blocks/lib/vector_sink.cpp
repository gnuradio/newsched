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

#include <gnuradio/scheduler.hpp>

using namespace std;
namespace gr {
namespace blocks {

template <class T>
vector_sink<T>::vector_sink(const size_t vlen, const size_t reserve_items)
    : sync_block("vector_sink"), d_vlen(vlen)
{
    d_data.reserve(d_vlen * reserve_items);
}

template <class T>
work_return_code_t vector_sink<T>::work(std::vector<block_work_input>& work_input,
                                        std::vector<block_work_output>& work_output)
{
    T* iptr = (T*)work_input[0].buffer->read_ptr();
    int noutput_items = work_input[0].n_items;

    for (unsigned int i = 0; i < noutput_items * d_vlen; i++)
        d_data.push_back(iptr[i]);

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