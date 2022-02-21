/* -*- c++ -*- */
/*
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "nop_cpu.hh"
#include "nop_cpu_gen.hh"

namespace gr {
namespace blocks {

nop_cpu::nop_cpu(block_args args) : INHERITED_CONSTRUCTORS {}

work_return_code_t nop_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                  std::vector<block_work_output_sptr>& work_output)
{
    produce_each(work_output[0]->n_items, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr