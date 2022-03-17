/* -*- c++ -*- */
/*
 * Copyright 2013,2014 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include "base.h"
#include <gnuradio/zeromq/req_source.h>

namespace gr {
namespace zeromq {

class req_source_cpu : public virtual req_source, public virtual base_source
{
public:
    req_source_cpu(block_args args);
    work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;
private:
    bool d_req_pending = false;
};

} // namespace zeromq
} // namespace gr