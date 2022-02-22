/* -*- c++ -*- */
/*
 * Copyright 2004,2008,2009,2013,2017-2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/blocks/vector_sink.h>

namespace gr {
namespace blocks {

template <class T>
class vector_sink_cpu : public vector_sink<T>
{
public:
    vector_sink_cpu(const typename vector_sink<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

    void on_parameter_query(param_action_sptr action) override
    {
        gr_log_debug(
            this->_debug_logger, "block {}: on_parameter_query param_id: {}", this->id(), action->id());
        pmtf::pmt param = d_data;
        // auto data = pmtf::get_as<std::vector<float>>(*param);
        action->set_pmt_value(param);
    }

protected:
    std::vector<T> d_data;
    std::vector<tag_t> d_tags;
    size_t d_vlen;
};


} // namespace blocks
} // namespace gr
