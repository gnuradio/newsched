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

#include <gnuradio/pdu/stream_to_pdu.h>

namespace gr {
namespace pdu {

class stream_to_pdu_cpu : public stream_to_pdu
{
public:
    stream_to_pdu_cpu(const typename stream_to_pdu::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    size_t d_packet_len;
};


} // namespace pdu
} // namespace gr
