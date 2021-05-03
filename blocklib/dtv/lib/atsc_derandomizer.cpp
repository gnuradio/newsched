/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "atsc_derandomizer.hpp"
#include "gnuradio/dtv/atsc_consts.hpp"

namespace gr {
namespace dtv {

atsc_derandomizer::sptr atsc_derandomizer::make()
{
    return std::make_shared<atsc_derandomizer>();
}

atsc_derandomizer::atsc_derandomizer() : gr::sync_block("dtv_atsc_derandomizer")
{
    add_port(
        port<uint8_t>::make("in", port_direction_t::INPUT, { ATSC_MPEG_PKT_LENGTH }));
    add_port(
        port<uint8_t>::make("out", port_direction_t::OUTPUT, { ATSC_MPEG_PKT_LENGTH }));

    add_port(untyped_port::make("plinfo", port_direction_t::INPUT, sizeof(plinfo)));

    d_rand.reset();
    set_tag_propagation_policy(tag_propagation_policy_t::TPP_DONT);
}

work_return_code_t atsc_derandomizer::work(std::vector<block_work_input>& work_input,
                                           std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const uint8_t*>(work_input[0].items());
    auto out = static_cast<uint8_t*>(work_output[0].items());
    auto plin = static_cast<const plinfo*>(work_input[1].items());
    auto noutput_items = work_output[0].n_items;

    for (int i = 0; i < noutput_items; i++) {
        assert(plin[i].regular_seg_p());

        if (plin[i].first_regular_seg_p())
            d_rand.reset();

        d_rand.derandomize(&out[i * ATSC_MPEG_PKT_LENGTH], &in[i * ATSC_MPEG_PKT_LENGTH]);

        // Check the pipeline info for error status and and set the
        // corresponding bit in transport packet header.

        if (plin[i].transport_error_p())
            out[i * ATSC_MPEG_PKT_LENGTH + 1] |= MPEG_TRANSPORT_ERROR_BIT;
        else
            out[i * ATSC_MPEG_PKT_LENGTH + 1] &= ~MPEG_TRANSPORT_ERROR_BIT;
    }

    produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}

} /* namespace dtv */
} /* namespace gr */
