/* -*- c++ -*- */
/*
 * Copyright 2019 Bastian Bloessl <mail@bastibl.net>.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "msg_forward_cpu.h"
#include "msg_forward_cpu_gen.h"

namespace gr {
namespace blocks {

msg_forward_cpu::msg_forward_cpu(block_args args) : INHERITED_CONSTRUCTORS, d_max_messages(args.max_messages) {}

work_return_code_t msg_forward_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                  std::vector<block_work_output_sptr>& work_output)
{
    // there are no work inputs or outputs -- not sure why this would need to get called
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr
