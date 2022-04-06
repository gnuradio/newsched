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

#include <gnuradio/streamops/probe_signal.h>

namespace gr {
namespace streamops {

template <class T>
class probe_signal_cpu : public probe_signal<T>
{
public:
    probe_signal_cpu(const typename probe_signal<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    // Declare private variables here
};


} // namespace streamops
} // namespace gr