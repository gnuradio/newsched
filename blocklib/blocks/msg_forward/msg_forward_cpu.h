/* -*- c++ -*- */
/*
 * Copyright 2019 Bastian Bloessl <mail@bastibl.net>.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <pmtf/string.hpp>
#include <gnuradio/blocks/msg_forward.h>

namespace gr {
namespace blocks {

class msg_forward_cpu : public msg_forward
{
public:
    msg_forward_cpu(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;
    // virtual size_t message_count() { return d_msg_cnt; }

protected:
    void handle_msg_in(pmtf::pmt msg)
    {

        // gr_log_info(
        //     _logger, "{} got message: {}", this->alias(), pmtf::string(msg).data());
        // GR_LOG_INFO(_logger, "got msg on block {}", alias());
        // d_msg_cnt++;

        size_t msg_cnt = pmtf::get_as<size_t>(*param_message_count);
        *param_message_count = ++msg_cnt;

        gr_log_debug(
            _debug_logger, "{}", msg_cnt);
        gr_log_info(
            _logger, "{}", msg_cnt);
        if (d_max_messages && msg_cnt >= d_max_messages)
        {
            input_message_port("system")->post("done");
        }
        get_message_port("out")->post(msg);
    }

private:
    size_t d_max_messages = 0;
    size_t d_msg_cnt = 0;
};

} // namespace blocks
} // namespace gr
