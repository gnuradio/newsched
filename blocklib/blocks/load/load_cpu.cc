/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "load_cpu.hh"
#include "load_cpu_gen.hh"

namespace gr {
namespace blocks {

load_cpu::load_cpu(block_args args) : INHERITED_CONSTRUCTORS, d_load(args.iterations) {}

work_return_code_t load_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                  std::vector<block_work_output_sptr>& work_output)
{
    auto iptr = work_input[0]->items<uint8_t>();
    auto optr = work_output[0]->items<uint8_t>();
    int size = work_output[0]->n_items * work_output[0]->buffer->item_size();

    // std::load(iptr, iptr + size, optr);
    for (size_t i=0; i < d_load; i++)
        memcpy(optr, iptr, size);

    work_output[0]->n_produced = work_output[0]->n_items;
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr