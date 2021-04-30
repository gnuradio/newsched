/* -*- c++ -*- */
/*
 * Copyright 2021 Perspecta Labs, Inc..
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../atsc_pnXXX.hpp"
#include <gnuradio/dtv/atsc_plinfo.hpp>
#include <gnuradio/dtv/cuda/atsc_equalizer_cuda.hpp>
#include <volk/volk.h>

extern void exec_filterN(
    float* in, float* out, float* taps, int ntaps, int nsamps, cudaStream_t stream);
void exec_adaptN(float* in,
                 float* out,
                 float* taps,
                 float* train,
                 int ntaps,
                 int nsamps,
                 cudaStream_t stream);

void exec_blockadaptN(float* in,
                      float* filtered,
                      float* taps,
                      float* train,
                      int ntaps,
                      int nsamps,
                      cudaStream_t stream);

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

atsc_equalizer_cuda::atsc_equalizer_cuda() : gr::block("dtv_atsc_equalizer")
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
    checkCudaErrors(
        cudaMalloc((void**)&d_dev_data_2, ATSC_DATA_SEGMENT_LENGTH * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_dev_taps, NTAPS * sizeof(float)));
    checkCudaErrors(cudaMemset(d_dev_taps, 0, NTAPS * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&data_mem,
                               (ATSC_DATA_SEGMENT_LENGTH + NTAPS) * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&data_mem2,
                               (ATSC_DATA_SEGMENT_LENGTH + NTAPS) * sizeof(float)));

    checkCudaErrors(
        cudaMalloc((void**)&d_dev_train1, KNOWN_FIELD_SYNC_LENGTH * sizeof(float)));
    checkCudaErrors(
        cudaMalloc((void**)&d_dev_train2, KNOWN_FIELD_SYNC_LENGTH * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_dev_train1,
                               training_sequence1,
                               sizeof(float) * KNOWN_FIELD_SYNC_LENGTH,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dev_train2,
                               training_sequence2,
                               sizeof(float) * KNOWN_FIELD_SYNC_LENGTH,
                               cudaMemcpyHostToDevice));

    cudaStreamCreate(&stream);
}

// void atsc_equalizer_cuda::adaptN(const float* input_samples,
//                             const float* training_pattern,
//                             float* output_samples,
//                             int nsamples)
// {
//     static const double BETA = 0.00005; // FIXME figure out what this ought to be
//                                         // FIXME add gear-shifting

//     for (int j = 0; j < nsamples; j++) {
//         output_samples[j] = 0;
//         volk_32f_x2_dot_prod_32f(
//             &output_samples[j], &input_samples[j], &d_taps[0], NTAPS);

//         float e = output_samples[j] - training_pattern[j];

//         // update taps...
//         float tmp_taps[NTAPS];
//         volk_32f_s32f_multiply_32f(tmp_taps, &input_samples[j], BETA * e, NTAPS);

//         // std::ofstream dbgfile6("/tmp/ns_taps_data6.bin",
//         //                        std::ios::app | std::ios::binary);
//         // dbgfile6.write((char*)tmp_taps, sizeof(float) * (NTAPS));

//         volk_32f_x2_subtract_32f(&d_taps[0], &d_taps[0], tmp_taps, NTAPS);
//     }

//     // std::ofstream dbgfile5("/tmp/ns_taps_data5.bin", std::ios::out |
//     std::ios::binary);
//     // dbgfile5.write((char*)output_samples, sizeof(float) * (nsamples));
// }

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
        checkCudaErrors(cudaMemset(&data_mem[0], 0, NPRETAPS * sizeof(float)));
        checkCudaErrors(cudaMemcpyAsync(&data_mem[NPRETAPS],
                                        in + i * ATSC_DATA_SEGMENT_LENGTH,
                                        ATSC_DATA_SEGMENT_LENGTH * sizeof(float),
                                        cudaMemcpyDeviceToDevice,
                                        stream));

        d_flags = plin[i].flags();
        d_segno = plin[i].segno();

        d_buff_not_filled = false;
        i++;
    }


    for (; i < noutput_items; i++) {

        checkCudaErrors(cudaMemcpyAsync(&data_mem[ATSC_DATA_SEGMENT_LENGTH + NPRETAPS],
                                        in + i * ATSC_DATA_SEGMENT_LENGTH,
                                        (NTAPS - NPRETAPS) * sizeof(float),
                                        cudaMemcpyDeviceToDevice,
                                        stream));

        // cudaStreamSynchronize(stream);

        if (d_segno == -1) {
            if (d_flags & 0x0010) {
                // float tmp_in[ATSC_DATA_SEGMENT_LENGTH + NTAPS];
                // float tmp_out[ATSC_DATA_SEGMENT_LENGTH];
                // float train[KNOWN_FIELD_SYNC_LENGTH];

                // checkCudaErrors(cudaMemcpy(tmp_in, data_mem, (ATSC_DATA_SEGMENT_LENGTH
                // + NTAPS)*sizeof(float), cudaMemcpyDeviceToHost));
                // checkCudaErrors(cudaMemcpy(train, d_dev_train2,
                // KNOWN_FIELD_SYNC_LENGTH*sizeof(float), cudaMemcpyDeviceToHost));

                // adaptN(tmp_in, train, tmp_out, KNOWN_FIELD_SYNC_LENGTH);

                // checkCudaErrors(cudaMemcpy(d_dev_taps, d_taps.data(),
                // NTAPS*sizeof(float), cudaMemcpyHostToDevice));

                // filter with the current weights
                // exec_filterN(data_mem,
                //              data_mem2,
                //              d_dev_taps,
                //              NTAPS,
                //              ATSC_DATA_SEGMENT_LENGTH,
                //              stream);
                // exec_blockadaptN(data_mem,
                //             data_mem2,
                //             d_dev_taps,
                //             d_dev_train2,
                //             NTAPS,
                //             KNOWN_FIELD_SYNC_LENGTH,
                //             stream);
                // update the weights
                exec_adaptN(data_mem,
                            d_dev_data_2,
                            d_dev_taps,
                            d_dev_train2,
                            NTAPS,
                            KNOWN_FIELD_SYNC_LENGTH,
                            stream);


            } else {

                // float tmp_in[ATSC_DATA_SEGMENT_LENGTH + NTAPS];
                // float tmp_out[ATSC_DATA_SEGMENT_LENGTH];
                // float train[KNOWN_FIELD_SYNC_LENGTH];

                // checkCudaErrors(cudaMemcpy(tmp_in, data_mem, (ATSC_DATA_SEGMENT_LENGTH
                // + NTAPS)*sizeof(float), cudaMemcpyDeviceToHost));
                // checkCudaErrors(cudaMemcpy(train, d_dev_train1,
                // KNOWN_FIELD_SYNC_LENGTH*sizeof(float), cudaMemcpyDeviceToHost));

                // adaptN(tmp_in, train, tmp_out, KNOWN_FIELD_SYNC_LENGTH);

                // checkCudaErrors(cudaMemcpy(d_dev_taps, d_taps.data(),
                // NTAPS*sizeof(float), cudaMemcpyHostToDevice));

                // exec_filterN(data_mem,
                //              data_mem2,
                //              d_dev_taps,
                //              NTAPS,
                //              ATSC_DATA_SEGMENT_LENGTH,
                //              stream);
                // exec_blockadaptN(data_mem,
                //             data_mem2,
                //             d_dev_taps,
                //             d_dev_train1,
                //             NTAPS,
                //             KNOWN_FIELD_SYNC_LENGTH,
                //             stream);
                exec_adaptN(data_mem,
                            d_dev_data_2,
                            d_dev_taps,
                            d_dev_train1,
                            NTAPS,
                            KNOWN_FIELD_SYNC_LENGTH,
                            stream);

            }
            checkCudaErrors(cudaPeekAtLastError());
            // cudaStreamSynchronize(stream);

        } else {

            exec_filterN(data_mem,
                         &out[output_produced * ATSC_DATA_SEGMENT_LENGTH],
                         d_dev_taps,
                         NTAPS,
                         ATSC_DATA_SEGMENT_LENGTH,
                         stream);
            checkCudaErrors(cudaPeekAtLastError());
            // cudaStreamSynchronize(stream);

            plout[output_produced++] = plinfo(d_flags, d_segno);
        }

        // cudaStreamSynchronize(stream);

        checkCudaErrors(cudaMemcpyAsync(&data_mem[0],
                                        &data_mem[ATSC_DATA_SEGMENT_LENGTH],
                                        NPRETAPS * sizeof(float),
                                        cudaMemcpyDeviceToDevice,
                                        stream));

        checkCudaErrors(cudaMemcpyAsync(&data_mem[NPRETAPS],
                                        in + i * ATSC_DATA_SEGMENT_LENGTH,
                                        ATSC_DATA_SEGMENT_LENGTH * sizeof(float),
                                        cudaMemcpyDeviceToDevice,
                                        stream));

        // cudaStreamSynchronize(stream);

        d_flags = plin[i].flags();
        d_segno = plin[i].segno();
    }

    cudaStreamSynchronize(stream);
    consume_each(noutput_items, work_input);
    produce_each(output_produced, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace dtv
} /* namespace gr */
