/* -*- c++ -*- */
/*
 * Copyright 2013,2014,2019 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include "base.h"
#include <gnuradio/zeromq/rep_sink.h>

namespace gr {
namespace zeromq {

class rep_sink_cpu : public virtual rep_sink, public virtual base_sink
{
public:
    rep_sink_cpu(block_args args);
    work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;
    std::string last_endpoint() const override { return base_sink::last_endpoint(); }
};

} // namespace zeromq
} // namespace gr