/* -*- c++ -*- */
/*
 * Copyright 2021 Perspecta Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/dtv/cuda/atsc_equalizer_cuda.hpp>
#include "../atsc_pnXXX.hpp"
#include <gnuradio/dtv/atsc_plinfo.hpp>
#include <volk/volk.h>

extern void exec_filterN(
    float* in, float* out, float* taps, int ntaps, int nsamps, cudaStream_t stream);
void exec_adaptN(
    float* in, float* out, float* taps, float* train, int ntaps, int nsamps, cudaStream_t stream);

namespace gr {
namespace dtv {

atsc_equalizer_cuda::sptr atsc_equalizer_cuda::make()
{
    return std::make_shared<atsc_equalizer_cuda>();
}

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

atsc_equalizer_cuda::atsc_equalizer_cuda()
    : gr::block("dtv_atsc_equalizer")
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

    const int alignment_multiple = volk_get_alignment() / sizeof(float);
    // set_alignment(std::max(1, alignment_multiple));
    set_output_multiple(std::max(1, alignment_multiple));


    checkCudaErrors(cudaMalloc((void**)&d_dev_data,
                               2 * (ATSC_DATA_SEGMENT_LENGTH + NTAPS) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_dev_data_2,
                               ATSC_DATA_SEGMENT_LENGTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_dev_taps, NTAPS * sizeof(float)));


    checkCudaErrors(cudaMalloc((void**)&d_dev_train1, KNOWN_FIELD_SYNC_LENGTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_dev_train2, KNOWN_FIELD_SYNC_LENGTH * sizeof(float)));

    checkCudaErrors(cudaMemcpy(
        d_dev_train1, training_sequence1, sizeof(float) * KNOWN_FIELD_SYNC_LENGTH, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_dev_train2, training_sequence2, sizeof(float) * KNOWN_FIELD_SYNC_LENGTH, cudaMemcpyHostToDevice));

    cudaStreamCreate(&stream);
}

work_return_code_t atsc_equalizer_cuda::work(std::vector<block_work_input>& work_input,
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

        cudaStreamSynchronize(stream);
        // std::cout << d_dev_data << " " << data_mem << std::endl;
        checkCudaErrors(cudaMemcpy(d_dev_data,
                                   &data_mem[0],
                                   sizeof(float) * (ATSC_DATA_SEGMENT_LENGTH + NTAPS),
                                   cudaMemcpyHostToDevice));

        if (d_segno == -1) {
            if (d_flags & 0x0010) {
                // adaptN(data_mem, training_sequence2, data_mem2, KNOWN_FIELD_SYNC_LENGTH);

                exec_adaptN(d_dev_data,
                            d_dev_data_2,
                            d_dev_taps,
                            d_dev_train2,
                            NTAPS,
                            KNOWN_FIELD_SYNC_LENGTH,
                            stream);
            } else { 
                // adaptN(data_mem, training_sequence1, data_mem2, KNOWN_FIELD_SYNC_LENGTH);
                
                exec_adaptN(d_dev_data,
                            d_dev_data_2,
                            d_dev_taps,
                            d_dev_train1,
                            NTAPS,
                            KNOWN_FIELD_SYNC_LENGTH,
                            stream);
                
            }
            cudaStreamSynchronize(stream);

            // d_nsamples = d_filter.set_taps(d_taps);

        } else {
            // filterN(data_mem, data_mem2, ATSC_DATA_SEGMENT_LENGTH);

            // checkCudaErrors(cudaMemcpy(
            //     d_dev_taps, &d_taps[0], sizeof(float) * NTAPS, cudaMemcpyHostToDevice));

            exec_filterN(d_dev_data,
                         d_dev_data,
                         d_dev_taps,
                         NTAPS,
                         ATSC_DATA_SEGMENT_LENGTH,
                         stream);
            cudaStreamSynchronize(stream);

            // float tmp[ATSC_DATA_SEGMENT_LENGTH];

            checkCudaErrors(cudaMemcpy(data_mem2,
                                       d_dev_data,
                                       sizeof(float) * (ATSC_DATA_SEGMENT_LENGTH),
                                       cudaMemcpyDeviceToHost));


            // d_filter.filter(ATSC_DATA_SEGMENT_LENGTH, data_mem + NPRETAPS,
            // data_mem2);


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


} // namespace dtv
} /* namespace gr */
