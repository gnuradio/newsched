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

#include <gnuradio/math/complex_to_mag_squared.hh>
#include <volk/volk.h>

namespace gr {
namespace math {

class complex_to_mag_squared_cpu : public complex_to_mag_squared
{
public:
    complex_to_mag_squared_cpu(const block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    size_t d_vlen;
};

} // namespace math
} // namespace gr