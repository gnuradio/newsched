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

#include <fstream>
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

    const int alignment_multiple = volk_get_alignment() / sizeof(float);
    // set_alignment(std::max(1, alignment_multiple));
    set_output_multiple(std::max(1, alignment_multiple));


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

#if 0 // standard lms
    for (int j = 0; j < nsamples; j++) {
        output_samples[j] = 0;
        volk_32f_x2_dot_prod_32f(
            &output_samples[j], &input_samples[j], &d_taps[0], NTAPS);

        float e = output_samples[j] - training_pattern[j];

        // update taps...
        float tmp_taps[NTAPS];
        volk_32f_s32f_multiply_32f(tmp_taps, &input_samples[j], BETA * e, NTAPS);

        // std::ofstream dbgfile6("/tmp/ns_taps_data6.bin",
        //                        std::ios::app | std::ios::binary);
        // dbgfile6.write((char*)tmp_taps, sizeof(float) * (NTAPS));

        volk_32f_x2_subtract_32f(&d_taps[0], &d_taps[0], tmp_taps, NTAPS);
    }

#else // block lms
    int block_size = nsamples; //NTAPS*8;
    int nblocks = nsamples / block_size;
    nsamples = block_size * nblocks;
    float e[block_size];

    for (int j = 0; j < nsamples; j += block_size) {
        
        for (int b = 0; b < block_size; b++) {

            output_samples[j + b] = 0;
            volk_32f_x2_dot_prod_32f(
                &output_samples[j+b], &input_samples[j+b], &d_taps[0], NTAPS);
            e[b] = output_samples[j+b] - training_pattern[j+b];
        }


        float f;
        volk_32f_x2_dot_prod_32f(
                &f, &input_samples[j], &e[0], block_size);

        // update taps...
        float tmp_taps[NTAPS];
        volk_32f_s32f_multiply_32f(tmp_taps, &input_samples[j], BETA * f, NTAPS);

        // std::ofstream dbgfile6("/tmp/ns_taps_data6.bin",
        //                        std::ios::app | std::ios::binary);
        // dbgfile6.write((char*)tmp_taps, sizeof(float) * (NTAPS));

        volk_32f_x2_subtract_32f(&d_taps[0], &d_taps[0], tmp_taps, NTAPS);
    }
#endif

    // std::ofstream dbgfile5("/tmp/ns_taps_data5.bin", std::ios::out | std::ios::binary);
    // dbgfile5.write((char*)output_samples, sizeof(float) * (nsamples));
}

work_return_code_t atsc_equalizer::work(std::vector<block_work_input>& work_input,
                                        std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const float*>(work_input[0].items());
    auto out = static_cast<float*>(work_output[0].items());
    auto plin = static_cast<const plinfo*>(work_input[1].items());
    auto plout = static_cast<plinfo*>(work_output[1].items());

    auto noutput_items = work_output[0].n_items;
    auto ninput_items = work_input[0].n_items;
    if (ninput_items < noutput_items) {
        return work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS;
    }

    int output_produced = 0;
    int i = 0;

    plinfo pli_in;
    if (d_buff_not_filled) {
        memset(&data_mem[0], 0, NPRETAPS * sizeof(float));
        memcpy(&data_mem[NPRETAPS],
               in + i * ATSC_DATA_SEGMENT_LENGTH,
               ATSC_DATA_SEGMENT_LENGTH * sizeof(float));

        d_flags = plin[i].flags();
        d_segno = plin[i].segno();

        d_buff_not_filled = false;
        i++;
    }

    for (; i < noutput_items; i++) {

        memcpy(&data_mem[ATSC_DATA_SEGMENT_LENGTH + NPRETAPS],
               in + i * ATSC_DATA_SEGMENT_LENGTH,
               (NTAPS - NPRETAPS) * sizeof(float));

        // std::ofstream dbgfile2("/tmp/ns_taps_data2.bin",
        //                        std::ios::out | std::ios::binary);
        // dbgfile2.write((char*)data_mem,
        //                sizeof(float) * (ATSC_DATA_SEGMENT_LENGTH + NTAPS));


        if (d_segno == -1) {
            if (d_flags & 0x0010) {
                adaptN(data_mem, training_sequence2, data_mem2, KNOWN_FIELD_SYNC_LENGTH);
            } else {
                adaptN(data_mem, training_sequence1, data_mem2, KNOWN_FIELD_SYNC_LENGTH);
            }

        // std::ofstream dbgfile7("/tmp/ns_taps_data7.bin",
        //                        std::ios::out | std::ios::binary);
        // dbgfile7.write((char*)data_mem,
        //                sizeof(float) * (ATSC_DATA_SEGMENT_LENGTH + NTAPS));

        //     std::ofstream dbgfile("/tmp/ns_taps_data1.bin",
        //                           std::ios::out | std::ios::binary);
        //     dbgfile.write((char*)d_taps.data(), d_taps.size() * sizeof(d_taps[0]));


        //     std::ofstream dbgfile3("/tmp/ns_taps_data3.bin",
        //                            std::ios::out | std::ios::binary);
        //     dbgfile3.write((char*)training_sequence1,
        //                    sizeof(float) * (KNOWN_FIELD_SYNC_LENGTH));

        //     std::ofstream dbgfile4("/tmp/ns_taps_data4.bin",
        //                            std::ios::out | std::ios::binary);
        //     dbgfile4.write((char*)training_sequence2,
        //                    sizeof(float) * (KNOWN_FIELD_SYNC_LENGTH));

        } else {
            filterN(data_mem, data_mem2, ATSC_DATA_SEGMENT_LENGTH);

            memcpy(&out[output_produced * ATSC_DATA_SEGMENT_LENGTH],
                   data_mem2,
                   ATSC_DATA_SEGMENT_LENGTH * sizeof(float));

            plout[output_produced++] = plinfo(d_flags, d_segno);
        }

        memcpy(data_mem, &data_mem[ATSC_DATA_SEGMENT_LENGTH], NPRETAPS * sizeof(float));
        memcpy(&data_mem[NPRETAPS],
               in + i * ATSC_DATA_SEGMENT_LENGTH,
               ATSC_DATA_SEGMENT_LENGTH * sizeof(float));

        d_flags = plin[i].flags();
        d_segno = plin[i].segno();
    }

    consume_each(noutput_items, work_input);
    produce_each(output_produced, work_output);
    return work_return_code_t::WORK_OK;
}

} /* namespace dtv */
} /* namespace gr */
