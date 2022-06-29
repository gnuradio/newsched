/* -*- c++ -*- */
/*
 * Copyright 2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/blocks/tags_strobe.h>

namespace gr {
namespace blocks {

class tags_strobe_cpu : public virtual tags_strobe
{
public:
    tags_strobe_cpu(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    uint64_t d_nsamps;
    tag_t d_tag;
    uint64_t d_offset = 0;
};

} // namespace blocks
} // namespace gr