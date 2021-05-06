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


#include <gnuradio/dtv/cuda/atsc_viterbi_decoder_cuda.hpp>
#include "../atsc_viterbi_mux.hpp"

#include <gnuradio/dtv/atsc_consts.hpp>
#include <gnuradio/dtv/atsc_plinfo.hpp>

extern void exec_deinterleave_kernel(const float* in, float* out, cudaStream_t stream);
extern void exec_viterbi_kernel(float* in,
                                unsigned char* out,
                                float* path_metrics,
                                unsigned long long* traceback,
                                int* post_coder_state,
                                cudaStream_t stream);


extern void
exec_interleave_kernel(unsigned char* in, unsigned char* out, cudaStream_t stream);

using namespace gr::dtv;

namespace gr {
namespace dtv {

atsc_viterbi_decoder_cuda::sptr atsc_viterbi_decoder_cuda::make()
{
    return std::make_shared<atsc_viterbi_decoder_cuda>();
}

atsc_viterbi_decoder_cuda::atsc_viterbi_decoder_cuda()
    : sync_block(
          "dtv_atsc_viterbi_decoder_cuda")
{
    add_port(
        port<float>::make("in", port_direction_t::INPUT, { ATSC_DATA_SEGMENT_LENGTH }));
    add_port(port<uint8_t>::make(
        "out", port_direction_t::OUTPUT, { ATSC_MPEG_RS_ENCODED_LENGTH }));

    add_port(untyped_port::make("plinfo", port_direction_t::INPUT, sizeof(plinfo)));
    add_port(untyped_port::make("plinfo", port_direction_t::OUTPUT, sizeof(plinfo)));


    set_output_multiple(NCODERS);


    /*
     * These fifo's handle the alignment problem caused by the
     * inherent decoding delay of the individual viterbi decoders.
     * The net result is that this entire block has a pipeline latency
     * of 12 complete segments.
     *
     * If anybody cares, it is possible to do it with less delay, but
     * this approach is at least somewhat understandable...
     */

    // the -4 is for the 4 sync symbols
    const int fifo_size = ATSC_DATA_SEGMENT_LENGTH - 4 - viterbi[0].delay();
    
    // unsigned int flags;
    // cudaGetDeviceFlags ( &flags );

    // std::cout << "flags: " << flags << std::endl;

    // checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleYield)); 

    // cudaGetDeviceFlags ( &flags );

    // std::cout << "flags: " << flags << std::endl;

    checkCudaErrors(
        cudaMalloc((void**)&d_data, sizeof(float) * NCODERS * ATSC_DATA_SEGMENT_LENGTH));

    checkCudaErrors(
        cudaMallocHost((void**)&d_host_in, sizeof(float) * NCODERS * ATSC_DATA_SEGMENT_LENGTH));

    checkCudaErrors(
        cudaMallocHost((void**)&d_host_out, OUTPUT_SIZE));        

    checkCudaErrors(cudaMalloc((void**)&d_dibits,
                               sizeof(unsigned char) * NCODERS * (enco_which_max + 797)));

    checkCudaErrors(
        cudaMalloc((void**)&d_out_copy, sizeof(unsigned char) * (OUTPUT_SIZE)));

    checkCudaErrors(cudaMalloc((void**)&d_path_metrics, sizeof(float) * NCODERS * 2 * 4));

    checkCudaErrors(cudaMemset(d_path_metrics, 0, sizeof(float) * NCODERS * 2 * 4));


    checkCudaErrors(
        cudaMalloc((void**)&d_traceback, sizeof(unsigned long long) * NCODERS * 2 * 4));

    checkCudaErrors(
        cudaMemset(d_traceback, 0, sizeof(unsigned long long) * NCODERS * 2 * 4));

    checkCudaErrors(
        cudaMalloc((void**)&d_post_coder_state, sizeof(unsigned char) * NCODERS));

    checkCudaErrors(cudaMemset(d_post_coder_state, 0, sizeof(unsigned char) * NCODERS));

    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

}

std::vector<float> atsc_viterbi_decoder_cuda::decoder_metrics() const
{
    std::vector<float> metrics(NCODERS);
    for (int i = 0; i < NCODERS; i++)
        metrics[i] = viterbi[i].best_state_metric();
    return metrics;
}

work_return_code_t atsc_viterbi_decoder_cuda::work(std::vector<block_work_input>& work_input,
                                   std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const float*>(work_input[0].items());
    auto out = static_cast<uint8_t*>(work_output[0].items());
    auto plin = static_cast<const plinfo*>(work_input[1].items());
    auto plout = static_cast<plinfo*>(work_output[1].items());

    auto noutput_items = work_output[0].n_items;
    assert(noutput_items % NCODERS == 0);
    

    unsigned char out_copy[OUTPUT_SIZE];
    float symbols[NCODERS][enco_which_max];
    unsigned char dibits[NCODERS][enco_which_max];
    unsigned char best_state[NCODERS][enco_which_max];

    // force it to be only 1 noutput item for now
    // noutput_items = NCODERS;
    for (int i = 0; i < noutput_items; i += NCODERS) {

        exec_deinterleave_kernel(in + i * ATSC_DATA_SEGMENT_LENGTH, d_data, streams[0]);      

        exec_viterbi_kernel(d_data,
                            d_dibits,
                            d_path_metrics,
                            d_traceback,
                            d_post_coder_state, streams[0]);
        
        exec_interleave_kernel(d_dibits, &out[i * ATSC_MPEG_RS_ENCODED_LENGTH], streams[0]);

        // checkCudaErrors(cudaMemcpyAsync(d_host_out,
        //                            &d_out_copy[0],
        //                            sizeof(unsigned char) * ATSC_MPEG_RS_ENCODED_LENGTH * NCODERS,
        //                            cudaMemcpyDeviceToHost, streams[0]));


        // copy output from contiguous temp buffer into final output
        for (int j = 0; j < NCODERS; j++) {
            plinfo::delay(plout[i + j], plin[i + j], NCODERS);
        }

        // cudaStreamSynchronize(streams[0]);
        // memcpy(&out[i * ATSC_MPEG_RS_ENCODED_LENGTH], d_host_out, OUTPUT_SIZE);

    }

    cudaStreamSynchronize(streams[0]);

    produce_each(noutput_items,work_output);
    return work_return_code_t::WORK_OK;
}

} /* namespace dtv */
} /* namespace gr */
