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

namespace gr {
namespace pdu {

template <class T>
stream_to_pdu_cpu<T>::stream_to_pdu_cpu(const typename stream_to_pdu<T>::block_args& args)
    : INHERITED_CONSTRUCTORS(T), d_packet_len(args.packet_len)
{
    this->set_output_multiple(args.packet_len);
}

template <class T>
work_return_code_t
stream_to_pdu_cpu<T>::work(std::vector<block_work_input_sptr>& work_input,
                           std::vector<block_work_output_sptr>& work_output)
{

    auto n_pdu = work_input[0]->n_items / d_packet_len;
    auto in = work_input[0]->items<T>();

    for (size_t n = 0; n < n_pdu; n++) {
        auto samples =
            pmtf::vector<T>(in + n * d_packet_len, in + (n + 1) * d_packet_len);
        auto d = pmtf::map({
            { "packet_len", d_packet_len },
        });

        auto pdu = pmtf::map({ { "data", samples }, { "meta", d } });

        this->get_message_port("pdus")->post(pdu);
    }

    this->consume_each(n_pdu * d_packet_len, work_input);
    return work_return_code_t::WORK_OK;
}

} /* namespace pdu */
} /* namespace gr */
