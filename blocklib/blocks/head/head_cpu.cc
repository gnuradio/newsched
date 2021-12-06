/* -*- c++ -*- */
/*
 * Copyright 2010,2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "head_cpu.hh"
#include "head_cpu_gen.hh"

namespace gr {
namespace blocks {

head_cpu::head_cpu(const block_args& args)
    : sync_block("head"), head(args), d_itemsize(args.itemsize), d_nitems(args.nitems)
{
}

work_return_code_t head_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                  std::vector<block_work_output_sptr>& work_output)
{
    auto iptr = work_input[0]->items<uint8_t>();
    auto optr = work_output[0]->items<uint8_t>();

    if (d_ncopied_items >= d_nitems) {
        work_output[0]->n_produced = 0;
        return work_return_code_t::WORK_DONE; // Done!
    }

    unsigned n = std::min(d_nitems - d_ncopied_items, (uint64_t)work_output[0]->n_items);

    if (n == 0) {
        work_output[0]->n_produced = 0;
        return work_return_code_t::WORK_OK;
    }

    memcpy(optr, iptr, n * work_input[0]->buffer->item_size());

    d_ncopied_items += n;
    work_output[0]->n_produced = n;

    return work_return_code_t::WORK_OK;
}

} /* namespace blocks */
} /* namespace gr */
