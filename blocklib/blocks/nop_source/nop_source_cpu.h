/* -*- c++ -*- */
/*
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/blocks/nop_source.h>

namespace gr {
namespace blocks {

class nop_source_cpu : public nop_source
{
public:
    nop_source_cpu(block_args args);
    work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;

protected:
    size_t d_nports;
};

} // namespace blocks
} // namespace gr