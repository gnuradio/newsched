/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/dtv/atsc_deinterleaver.hpp>
#include <gnuradio/dtv/atsc_consts.hpp>
#include <gnuradio/dtv/atsc_plinfo.hpp>
namespace gr {
namespace dtv {

atsc_deinterleaver::sptr atsc_deinterleaver::make()
{
    return std::make_shared<atsc_deinterleaver>();
}

atsc_deinterleaver::atsc_deinterleaver()
    : gr::sync_block(
          "atsc_deinterleaver"),
      alignment_fifo(156)
{
    add_port(
        port<uint8_t>::make("in", port_direction_t::INPUT, { ATSC_MPEG_RS_ENCODED_LENGTH }));
    add_port(
        port<uint8_t>::make("out", port_direction_t::OUTPUT, { ATSC_MPEG_RS_ENCODED_LENGTH }));

    add_port(untyped_port::make("plinfo", port_direction_t::INPUT, sizeof(plinfo)));
    add_port(untyped_port::make("plinfo", port_direction_t::OUTPUT, sizeof(plinfo)));

    m_fifo.reserve(s_interleavers);

    for (int i = 0; i < s_interleavers; i++)
        m_fifo.emplace_back((s_interleavers - 1 - i) * 4);

    sync();

    set_tag_propagation_policy(tag_propagation_policy_t::TPP_CUSTOM);
}

atsc_deinterleaver::~atsc_deinterleaver() {}

void atsc_deinterleaver::reset()
{
    sync();

    for (auto& i : m_fifo)
        i.reset();
}

work_return_code_t atsc_deinterleaver::work(std::vector<block_work_input>& work_input,
                                        std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const uint8_t*>(work_input[0].items());
    auto out = static_cast<uint8_t*>(work_output[0].items());
    auto plin = static_cast<const plinfo*>(work_input[1].items());
    auto plout = static_cast<plinfo*>(work_output[1].items());
    auto noutput_items = work_output[0].n_items;

    for (int i = 0; i < noutput_items; i++) {
        assert(plin[i].regular_seg_p());

        // reset commutator if required using INPUT pipeline info
        if (plin[i].first_regular_seg_p())
            sync();

        // remap OUTPUT pipeline info to reflect all data segment end-to-end delay
        plinfo::delay(plout[i], plin[i], s_interleavers);

        // now do the actual deinterleaving
        for (unsigned int j = 0; j < ATSC_MPEG_RS_ENCODED_LENGTH; j++) {
            out[i * ATSC_MPEG_RS_ENCODED_LENGTH + j] =
                alignment_fifo.stuff(transform(in[i * ATSC_MPEG_RS_ENCODED_LENGTH + j]));
        }
    }

    produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}

} /* namespace dtv */
} /* namespace gr */
