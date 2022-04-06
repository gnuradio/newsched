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

#include <gnuradio/streamops/probe_signal_v.h>

namespace gr {
namespace streamops {

template <class T>
class probe_signal_v_cpu : public probe_signal_v<T>
{
public:
    probe_signal_v_cpu(const typename probe_signal_v<T>::block_args& args);

    virtual work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;

private:
    // Just work with the private member variable, and pass it out as pmt when queried
    void on_parameter_query(param_action_sptr action) override
    {
        pmtf::pmt param = d_level;
        action->set_pmt_value(param);
    }

    size_t d_vlen;
    std::vector<T> d_level;
};


} // namespace streamops
} // namespace gr
