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

#include "atsc_pnXXX.hpp"
#include "atsc_syminfo.hpp"
#include "atsc_types.hpp"
#include <gnuradio/dtv/atsc_consts.hpp>
#include <gnuradio/dtv/atsc_equalizer.hpp>
#include <volk/volk.h>

#include <pmt/pmtf_scalar.hpp>
#include <pmt/pmtf_string.hpp>

namespace gr {
namespace dtv {

atsc_equalizer::sptr atsc_equalizer::make() { return std::make_shared<atsc_equalizer>(); }

static float bin_map(int bit) { return bit ? +5 : -5; }

static void init_field_sync_common(float* p, int mask)
{
    int i = 0;

    p[i++] = bin_map(1); // data segment sync pulse
    p[i++] = bin_map(0);
    p[i++] = bin_map(0);
    p[i++] = bin_map(1);

    for (int j = 0; j < 511; j++) // PN511
        p[i++] = bin_map(atsc_pn511[j]);

    for (int j = 0; j < 63; j++) // PN63
        p[i++] = bin_map(atsc_pn63[j]);

    for (int j = 0; j < 63; j++) // PN63, toggled on field 2
        p[i++] = bin_map(atsc_pn63[j] ^ mask);

    for (int j = 0; j < 63; j++) // PN63
        p[i++] = bin_map(atsc_pn63[j]);
}

atsc_equalizer::atsc_equalizer() : gr::block("dtv_atsc_equalizer")
{
    add_port(
        port<float>::make("in", port_direction_t::INPUT, { ATSC_DATA_SEGMENT_LENGTH }));
    add_port(
        port<float>::make("out", port_direction_t::OUTPUT, { ATSC_DATA_SEGMENT_LENGTH }));

    add_port(untyped_port::make("plinfo", port_direction_t::INPUT, sizeof(plinfo)));
    add_port(untyped_port::make("plinfo", port_direction_t::OUTPUT, sizeof(plinfo)));

    init_field_sync_common(training_sequence1, 0);
    init_field_sync_common(training_sequence2, 1);

    d_taps.resize(NTAPS, 0.0f);

    d_buff_not_filled = true;

    // const int alignment_multiple = volk_get_alignment() / sizeof(float);
    // set_alignment(std::max(1, alignment_multiple));

    set_tag_propagation_policy(
        tag_propagation_policy_t::TPP_CUSTOM); // use manual tag propagation
}

std::vector<float> atsc_equalizer::taps() const { return d_taps; }

std::vector<float> atsc_equalizer::data() const
{
    std::vector<float> ret(&data_mem2[0], &data_mem2[ATSC_DATA_SEGMENT_LENGTH - 1]);
    return ret;
}

void atsc_equalizer::filterN(const float* input_samples,
                             float* output_samples,
                             int nsamples)
{
    for (int j = 0; j < nsamples; j++) {
        output_samples[j] = 0;
        volk_32f_x2_dot_prod_32f(
            &output_samples[j], &input_samples[j], &d_taps[0], NTAPS);
    }
}

void atsc_equalizer::adaptN(const float* input_samples,
                            const float* training_pattern,
                            float* output_samples,
                            int nsamples)
{
    static const double BETA = 0.00005; // FIXME figure out what this ought to be
                                        // FIXME add gear-shifting

    for (int j = 0; j < nsamples; j++) {
        output_samples[j] = 0;
        volk_32f_x2_dot_prod_32f(
            &output_samples[j], &input_samples[j], &d_taps[0], NTAPS);

        float e = output_samples[j] - training_pattern[j];

        // update taps...
        float tmp_taps[NTAPS];
        volk_32f_s32f_multiply_32f(tmp_taps, &input_samples[j], BETA * e, NTAPS);
        volk_32f_x2_subtract_32f(&d_taps[0], &d_taps[0], tmp_taps, NTAPS);
    }
}

work_return_code_t atsc_equalizer::work(std::vector<block_work_input>& work_input,
                                        std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const float*>(work_input[0].items());
    auto out = static_cast<float*>(work_output[0].items());
    auto plin = static_cast<const plinfo*>(work_input[1].items());
    auto plout = static_cast<plinfo*>(work_output[1].items());

    auto noutput_items = work_output[0].n_items;

    int output_produced = 0;
    int i = 0;

    std::vector<tag_t> tags;
    auto tag_pmt = pmtf::pmt_string::make("plinfo");

    plinfo pli_in;
    if (d_buff_not_filled) {
        memset(&data_mem[0], 0, NPRETAPS * sizeof(float));
        memcpy(&data_mem[NPRETAPS],
               in + i * ATSC_DATA_SEGMENT_LENGTH,
               ATSC_DATA_SEGMENT_LENGTH * sizeof(float));

        // get_tags_in_window(tags, 0, 0, 1, tag_pmt);

        // if (tags.size() > 0) {
        //     pli_in.from_tag_value(pmt::to_uint64(tags[0].value));
        //     d_flags = pli_in.flags();
        //     d_segno = pli_in.segno();
        // } else {
        //     throw std::runtime_error("Atsc Equalizer: Plinfo Tag not found on sample");
        // }

        d_flags = plin[i].flags();
        d_segno = plin[i].segno();

        d_buff_not_filled = false;
        i++;
    }

    for (; i < noutput_items; i++) {

        memcpy(&data_mem[ATSC_DATA_SEGMENT_LENGTH + NPRETAPS],
               in + i * ATSC_DATA_SEGMENT_LENGTH,
               (NTAPS - NPRETAPS) * sizeof(float));

        if (d_segno == -1) {
            if (d_flags & 0x0010) {
                adaptN(data_mem, training_sequence2, data_mem2, KNOWN_FIELD_SYNC_LENGTH);
            } else if (!(d_flags & 0x0010)) {
                adaptN(data_mem, training_sequence1, data_mem2, KNOWN_FIELD_SYNC_LENGTH);
            }
        } else {
            filterN(data_mem, data_mem2, ATSC_DATA_SEGMENT_LENGTH);

            memcpy(&out[output_produced * ATSC_DATA_SEGMENT_LENGTH],
                   data_mem2,
                   ATSC_DATA_SEGMENT_LENGTH * sizeof(float));

            plinfo pli_out(d_flags, d_segno);
            // add_item_tag(0,
            //              nitems_written(0) + output_produced,
            //              tag_pmt,
            //              pmt::from_uint64(pli_out.get_tag_value()));
            plout[output_produced] = pli_out;

            output_produced++;
        }

        memcpy(data_mem, &data_mem[ATSC_DATA_SEGMENT_LENGTH], NPRETAPS * sizeof(float));
        memcpy(&data_mem[NPRETAPS],
               in + i * ATSC_DATA_SEGMENT_LENGTH,
               ATSC_DATA_SEGMENT_LENGTH * sizeof(float));

        // get_tags_in_window(tags, 0, i, i + 1, tag_pmt);
        // if (tags.size() > 0) {
        //     pli_in.from_tag_value(pmt::to_uint64(tags[0].value));
        //     d_flags = pli_in.flags();
        //     d_segno = pli_in.segno();
        // } else {
        //     throw std::runtime_error("Atsc Equalizer: Plinfo Tag not found on sample");
        // }
        d_flags = plin[i].flags();
        d_segno = plin[i].segno();
    }

    consume_each(noutput_items, work_input);
    produce_each(output_produced, work_output);
    return work_return_code_t::WORK_OK;
}

} /* namespace dtv */
} /* namespace gr */
