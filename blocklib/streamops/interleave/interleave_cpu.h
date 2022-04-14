/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/streamops/interleave.h>

namespace gr {
namespace streamops {

class interleave_cpu : public virtual interleave
{
public:
    interleave_cpu(block_args args);
    virtual work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;

private:
    const unsigned int d_ninputs;
    const unsigned int d_blocksize;
    size_t d_itemsize;
};

} // namespace streamops
} // namespace gr