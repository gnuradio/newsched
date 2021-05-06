/* -*- c++ -*- */
/*
 * Copyright 2014,2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "atsc_pnXXX.hpp"
#include "atsc_syminfo.hpp"
#include "atsc_types.hpp"
#include "gnuradio/dtv/atsc_consts.hpp"
#include <gnuradio/dtv/atsc_fs_checker.hpp>
#include <pmt/pmtf_scalar.hpp>
#include <pmt/pmtf_string.hpp>
#include <string>

#define ATSC_SEGMENTS_PER_DATA_FIELD 313

static const int PN511_ERROR_LIMIT = 20; // max number of bits wrong
static const int PN63_ERROR_LIMIT = 5;

namespace gr {
namespace dtv {

atsc_fs_checker::sptr atsc_fs_checker::make()
{
    return std::make_shared<atsc_fs_checker>();
}

atsc_fs_checker::atsc_fs_checker() : gr::block("dtv_atsc_fs_checker")
{
    add_port(
        port<float>::make("in", port_direction_t::INPUT, { ATSC_DATA_SEGMENT_LENGTH }));
    add_port(
        port<float>::make("out", port_direction_t::OUTPUT, { ATSC_DATA_SEGMENT_LENGTH }));
    add_port(
        untyped_port::make("plinfo", port_direction_t::OUTPUT, sizeof(plinfo)));

    reset();
}

void atsc_fs_checker::reset()
{
    d_index = 0;
    memset(d_sample_sr, 0, sizeof(d_sample_sr));
    memset(d_tag_sr, 0, sizeof(d_tag_sr));
    memset(d_bit_sr, 0, sizeof(d_bit_sr));
    d_field_num = 0;
    d_segment_num = 0;
}

work_return_code_t atsc_fs_checker::work(std::vector<block_work_input>& work_input,
                                         std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const float*>(work_input[0].items());
    auto out = static_cast<float*>(work_output[0].items());
    auto plout = static_cast<plinfo*>(work_output[1].items());
    auto noutput_items = work_output[0].n_items;
    auto ninput_items = work_input[0].n_items;

    // Need to figure out how to handle this more gracefully
    // The scheduler (currently) has no information about what the block
    // is doing and doesn't know to give ninput >= noutput
    if (ninput_items < noutput_items)
    {
        return work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS;
    }

    // std::cout << noutput_items << "/" << ninput_items << std::endl;

    int output_produced = 0;

    for (int i = 0; i < noutput_items; i++) {
        // check for a hit on the PN 511 pattern
        int errors = 0;

        for (int j = 0; j < LENGTH_511 && errors < PN511_ERROR_LIMIT; j++)
            errors +=
                (in[i * ATSC_DATA_SEGMENT_LENGTH + j + OFFSET_511] >= 0) ^ atsc_pn511[j];

        GR_LOG_DEBUG(_debug_logger,
                     std::string("second PN63 error count = ") + std::to_string(errors));

        if (errors < PN511_ERROR_LIMIT) { // 511 pattern is good.
            // determine if this is field 1 or field 2
            errors = 0;
            for (int j = 0; j < LENGTH_2ND_63; j++)
                errors += (in[i * ATSC_DATA_SEGMENT_LENGTH + j + OFFSET_2ND_63] >= 0) ^
                          atsc_pn63[j];

            // we should have either field 1 (== PN63) or field 2 (== ~PN63)
            if (errors <= PN63_ERROR_LIMIT) {
                GR_LOG_DEBUG(_debug_logger, "Found FIELD_SYNC_1")
                d_field_num = 1;    // We are in field number 1 now
                d_segment_num = -1; // This is the first segment
            } else if (errors >= (LENGTH_2ND_63 - PN63_ERROR_LIMIT)) {
                GR_LOG_DEBUG(_debug_logger, "Found FIELD_SYNC_2")
                d_field_num = 2;    // We are in field number 2 now
                d_segment_num = -1; // This is the first segment
            } else {
                // should be extremely rare.
                GR_LOG_WARN(_logger,
                            std::string("PN63 error count = ") + std::to_string(errors));
            }
        }

        if (d_field_num == 1 || d_field_num == 2) { // If we have sync
            // So we copy out current packet data to an output packet and fill its plinfo
            memcpy(&out[output_produced * ATSC_DATA_SEGMENT_LENGTH],
                   &in[i * ATSC_DATA_SEGMENT_LENGTH],
                   ATSC_DATA_SEGMENT_LENGTH * sizeof(float));

            plinfo pli_out;
            pli_out.set_regular_seg((d_field_num == 2), d_segment_num);

            d_segment_num++;
            if (d_segment_num > (ATSC_SEGMENTS_PER_DATA_FIELD - 1)) {
                d_field_num = 0;
                d_segment_num = 0;
            } else {
                // work_output[0].add_tag(
                //     work_output[0].nitems_written() + output_produced,
                //     tag_pmt,
                //     pmtf::pmt_scalar<uint32_t>::make(pli_out.get_tag_value()));
                plout[output_produced++] = pli_out;                
            }
        }
    }

    consume_each(noutput_items,work_input);
    produce_each(output_produced,work_output);
    return work_return_code_t::WORK_OK;
}

} /* namespace dtv */
} /* namespace gr */
