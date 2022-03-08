/* -*- c++ -*- */
/*
 * Copyright <COPYRIGHT_YEAR> <COPYRIGHT_AUTHOR>
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "newblock_cpu.h"
#include "newblock_cpu_gen.h"

namespace gr {
namespace newmod {

newblock_cpu::newblock_cpu(block_args args) : INHERITED_CONSTRUCTORS {}

work_return_code_t newblock_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                      std::vector<block_work_output_sptr>& work_output)
{
    // Do <+signal processing+>
    // Block specific code goes here
    return work_return_code_t::WORK_OK;
}


} // namespace newmod
} // namespace gr