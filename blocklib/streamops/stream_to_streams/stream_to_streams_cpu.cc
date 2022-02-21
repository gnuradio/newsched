/* -*- c++ -*- */
/*
 * Copyright 2012 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "stream_to_streams_cpu.hh"
#include "stream_to_streams_cpu_gen.hh"
#include <volk/volk.h>

namespace gr {
namespace streamops {

stream_to_streams_cpu::stream_to_streams_cpu(const block_args& args)
    : INHERITED_CONSTRUCTORS
{
}

work_return_code_t
stream_to_streams_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                       std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0]->items<uint8_t>();

    uint8_t * in_ptr = const_cast<uint8_t*>(in);
    auto noutput_items = work_output[0]->n_items;
    auto ninput_items = work_input[0]->n_items;
    size_t nstreams = work_output.size();

    auto total_items = std::min(ninput_items / nstreams, (size_t)noutput_items);
    auto itemsize = work_output[0]->buffer->item_size();

    for (size_t i = 0; i < total_items; i++) {
        for (size_t j = 0; j < nstreams; j++) {
            memcpy(work_output[j]->items<uint8_t>()+i*itemsize, in_ptr, itemsize);
            in_ptr += itemsize;
        }
    }

    produce_each(total_items, work_output);
    consume_each(total_items*nstreams, work_input);
    return work_return_code_t::WORK_OK;
}


} /* namespace streamops */
} /* namespace gr */
