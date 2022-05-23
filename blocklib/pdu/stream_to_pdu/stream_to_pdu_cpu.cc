/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "stream_to_pdu_cpu.h"
#include "stream_to_pdu_cpu_gen.h"

#include <gnuradio/pdu.h>

namespace gr {
namespace pdu {

stream_to_pdu_cpu::stream_to_pdu_cpu(const typename stream_to_pdu::block_args& args)
    : INHERITED_CONSTRUCTORS, d_packet_len(args.packet_len)
{
    this->set_output_multiple(args.packet_len);
}

work_return_code_t
stream_to_pdu_cpu::work(std::vector<block_work_input_sptr>& work_input,
                        std::vector<block_work_output_sptr>& work_output)
{

    auto n_pdu = work_input[0]->n_items / d_packet_len;
    auto in = work_input[0]->items<uint8_t>();
    int itemsize = work_input[0]->buffer->item_size();

    for (size_t n = 0; n < n_pdu; n++) {
        auto pdu_out =
            pmtf::pdu((void *)(in + n * d_packet_len * itemsize), d_packet_len * itemsize);

        pdu_out["packet_len"] = d_packet_len;

        get_message_port("pdus")->post(pdu_out);
    }

    consume_each(n_pdu * d_packet_len, work_input);
    return work_return_code_t::WORK_OK;
}

} /* namespace pdu */
} /* namespace gr */
