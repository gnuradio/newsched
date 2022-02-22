/* -*- c++ -*- */
/*
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "null_source_cpu.h"
#include "null_source_cpu_gen.h"

namespace gr {
namespace blocks {

null_source_cpu::null_source_cpu(block_args args) : INHERITED_CONSTRUCTORS {}

work_return_code_t null_source_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                         std::vector<block_work_output_sptr>& work_output)
{
    void* optr;
    auto itemsize = work_output[0]->buffer->item_size();
    for (size_t n = 0; n < work_output.size(); n++) {
        optr = work_output[n]->items<void>();
        auto noutput_items = work_output[n]->n_items;
        memset(optr, 0, noutput_items * itemsize);
        work_output[n]->n_produced = noutput_items;
    }

    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr