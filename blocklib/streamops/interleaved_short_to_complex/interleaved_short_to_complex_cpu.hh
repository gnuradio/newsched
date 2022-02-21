/* -*- c++ -*- */
/*
 * Copyright 2012 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/streamops/interleaved_short_to_complex.hh>

namespace gr {
namespace streamops {

class interleaved_short_to_complex_cpu : public interleaved_short_to_complex
{
public:
    interleaved_short_to_complex_cpu(const block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;


    void set_swap(bool swap);
    void set_scale_factor(float new_value);

private:
    float d_scalar;
    bool d_swap;
};


} // namespace streamops
} // namespace gr
