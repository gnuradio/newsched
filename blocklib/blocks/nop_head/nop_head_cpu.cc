/* -*- c++ -*- */
/*
 * Copyright 2010,2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "nop_head_cpu.hh"

namespace gr {
namespace blocks {

nop_head::sptr nop_head::make_cpu(const block_args& args)
{
    return std::make_shared<nop_head_cpu>(args);
}

nop_head_cpu::nop_head_cpu(const block_args& args)
    : nop_head(args), d_itemsize(args.itemsize), d_nitems(args.nitems)
{
}

work_return_code_t nop_head_cpu::work(std::vector<block_work_input>& work_input,
                                  std::vector<block_work_output>& work_output)
{
    auto* iptr = (uint8_t*)work_input[0].items();
    auto* optr = (uint8_t*)work_output[0].items();

    if (d_ncopied_items >= d_nitems) {
        work_output[0].n_produced = 0;
        return work_return_code_t::WORK_DONE; // Done!
    }

    unsigned n = std::min(d_nitems - d_ncopied_items, (uint64_t)work_output[0].n_items);

    if (n == 0) {
        work_output[0].n_produced = 0;
        return work_return_code_t::WORK_OK;
    }

    // memcpy(optr, iptr, n * d_itemsize);

    d_ncopied_items += n;
    work_output[0].n_produced = n;

    return work_return_code_t::WORK_OK;
}

} /* namespace blocks */
} /* namespace gr */
