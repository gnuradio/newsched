/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "vector_sink_cpu.hh"
#include <volk/volk.h>

namespace gr {
namespace blocks {

template <class T>
typename vector_sink<T>::sptr vector_sink<T>::make_cpu(const block_args& args)
{
    return std::make_shared<vector_sink_cpu<T>>(args);
}

template <class T>
vector_sink_cpu<T>::vector_sink_cpu(const typename vector_sink<T>::block_args& args)
    : sync_block("vector_sink"), vector_sink<T>(args), d_vlen(args.vlen)
{
    d_data.reserve(d_vlen * args.reserve_items);
}

template <class T>
work_return_code_t vector_sink_cpu<T>::work(std::vector<block_work_input>& work_input,
                                        std::vector<block_work_output>& work_output)
{
    auto iptr = work_input[0].items<T>();
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
