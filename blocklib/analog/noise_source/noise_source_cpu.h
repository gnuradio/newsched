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

#include <gnuradio/analog/noise_source.h>
#include <gnuradio/random.h>

namespace gr {
namespace analog {

template <class T>
class noise_source_cpu : public noise_source<T>
{
public:
    noise_source_cpu(const typename noise_source<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    gr::random d_rng;
};


} // namespace analog
} // namespace gr
