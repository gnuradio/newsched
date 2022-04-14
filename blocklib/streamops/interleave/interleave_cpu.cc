/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "interleave_cpu.h"
#include "interleave_cpu_gen.h"

namespace gr {
namespace streamops {

interleave_cpu::interleave_cpu(block_args args)
    : INHERITED_CONSTRUCTORS, d_ninputs(args.nstreams), d_blocksize(args.blocksize), d_itemsize(args.itemsize)

{
    set_relative_rate(d_ninputs);
    set_output_multiple(d_blocksize * d_ninputs);
}

work_return_code_t interleave_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                        std::vector<block_work_output_sptr>& work_output)
{

    // Since itemsize can be set after construction
    if (d_itemsize == 0) {
        d_itemsize = work_input[0]->buffer->item_size();
        return work_return_code_t::WORK_OK;
    }

    // Forecasting
    auto ninput_items = block_work_input::min_n_items(work_input);
    auto noutput_items = work_output[0]->n_items;
    auto noutput_blocks = std::min(ninput_items / d_blocksize, noutput_items / (d_blocksize * d_ninputs));  
    
    if (noutput_blocks < 1) {
        return work_return_code_t::WORK_INSUFFICIENT_OUTPUT_ITEMS;
    }

    auto out = work_output[0]->items<uint8_t>();

    for (unsigned int i = 0; i < noutput_blocks; i++) {
        for (unsigned int n = 0; n < d_ninputs; n++) {
            auto in = work_input[n]->items<uint8_t>() + d_itemsize * d_blocksize * i;
            memcpy(out, in, d_itemsize * d_blocksize);
            out += d_itemsize * d_blocksize;
        }
    }
    consume_each(noutput_blocks * d_blocksize, work_input);
    produce_each(noutput_blocks * d_blocksize * d_ninputs, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace streamops
} // namespace gr