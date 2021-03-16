/* -*- c++ -*- */
/*
 * Copyright 2021 gr-dtv author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/dtv/cuda/atsc_fs_checker_cuda.hpp>
#include "../atsc_pnXXX.hpp"
#include <gnuradio/dtv/atsc_syminfo.hpp>
#include <gnuradio/dtv/atsc_consts.hpp>
#include <gnuradio/dtv/atsc_plinfo.hpp>
#include <string>

#define ATSC_SEGMENTS_PER_DATA_FIELD 313

static const int PN511_ERROR_LIMIT = 20; // max number of bits wrong
static const int PN63_ERROR_LIMIT = 5;
void exec_atsc_fs_checker(float* in,
                          uint8_t* pn_seq,
                          int pn_len,
                          int offset,
                          int nitems,
                          uint16_t* nerrors,
                          cudaStream_t str);

using namespace gr::dtv;

namespace gr {
namespace dtv {

atsc_fs_checker_cuda::sptr atsc_fs_checker_cuda::make()
{
    return std::make_shared<atsc_fs_checker_cuda>();
}

atsc_fs_checker_cuda::atsc_fs_checker_cuda()
    : gr::block("dtv_atsc_fs_checker")
{
    add_port(
        port<float>::make("in", port_direction_t::INPUT, { ATSC_DATA_SEGMENT_LENGTH }));
    add_port(
        port<float>::make("out", port_direction_t::OUTPUT, { ATSC_DATA_SEGMENT_LENGTH }));
    add_port(
        untyped_port::make("plinfo", port_direction_t::OUTPUT, sizeof(plinfo)));

    
    reset();

    checkCudaErrors(
        cudaMallocHost((void**)&d_host_in,
                       d_max_output_items * ATSC_DATA_SEGMENT_LENGTH * sizeof(float)));
    checkCudaErrors(
        cudaMalloc((void**)&d_dev_in,
                   d_max_output_items * ATSC_DATA_SEGMENT_LENGTH * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&d_dev_atsc_pn511, LENGTH_511 * sizeof(uint8_t)));
    checkCudaErrors(
        cudaMalloc((void**)&d_dev_atsc_pn63, LENGTH_2ND_63 * sizeof(uint8_t)));
    checkCudaErrors(
        cudaMalloc((void**)&d_dev_nerrors511, d_max_output_items * sizeof(uint16_t)));
    checkCudaErrors(
        cudaMalloc((void**)&d_dev_nerrors63, d_max_output_items * sizeof(uint16_t)));
    checkCudaErrors(cudaMallocHost((void**)&d_host_nerrors511,
                                   d_max_output_items * sizeof(uint16_t)));
    checkCudaErrors(
        cudaMallocHost((void**)&d_host_nerrors63, d_max_output_items * sizeof(uint16_t)));

    checkCudaErrors(cudaMemcpy(d_dev_atsc_pn511,
                               atsc_pn511,
                               sizeof(uint8_t) * LENGTH_511,
                               cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_dev_atsc_pn63,
                               atsc_pn63,
                               sizeof(uint8_t) * LENGTH_2ND_63,
                               cudaMemcpyHostToDevice));

    set_output_multiple(d_max_output_items);
    // init_atsc_fs_checker();

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
}

void atsc_fs_checker_cuda::reset()
{
    d_index = 0;
    d_field_num = 0;
    d_segment_num = 0;
}

work_return_code_t atsc_fs_checker_cuda::work(std::vector<block_work_input>& work_input,
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

    int output_produced = 0;

    for (int i = 0; i < noutput_items; i += d_max_output_items) {

        size_t items_to_process = ((noutput_items - i) <= d_max_output_items)
                                      ? (noutput_items - i)
                                      : d_max_output_items;

        memcpy(d_host_in,
               in + i * ATSC_DATA_SEGMENT_LENGTH,
               items_to_process * sizeof(float) * ATSC_DATA_SEGMENT_LENGTH);

        checkCudaErrors(
            cudaMemcpyAsync(d_dev_in,
                            d_host_in,
                            items_to_process * sizeof(float) * ATSC_DATA_SEGMENT_LENGTH,
                            cudaMemcpyHostToDevice,
                            stream1));

        exec_atsc_fs_checker(d_dev_in,
                             d_dev_atsc_pn511,
                             LENGTH_511,
                             OFFSET_511,
                             items_to_process,
                             d_dev_nerrors511,
                             stream1);

        exec_atsc_fs_checker(d_dev_in,
                             d_dev_atsc_pn63,
                             LENGTH_2ND_63,
                             OFFSET_2ND_63,
                             items_to_process,
                             d_dev_nerrors63,
                             stream1);

        checkCudaErrors(cudaMemcpyAsync(d_host_nerrors511,
                                        d_dev_nerrors511,
                                        sizeof(int16_t) * items_to_process,
                                        cudaMemcpyDeviceToHost,
                                        stream1));

        checkCudaErrors(cudaMemcpyAsync(d_host_nerrors63,
                                        // checkCudaErrors(cudaMemcpy(d_host_nerrors63,
                                        d_dev_nerrors63,
                                        sizeof(int16_t) * items_to_process,
                                        cudaMemcpyDeviceToHost,
                                        stream1));

        cudaStreamSynchronize(stream1);

        for (int j = 0; j < items_to_process; j++) {
            int errors1 = d_host_nerrors511[j]; // needs to be the sum of errors across
                                                // the output multiple
            if (errors1 < PN511_ERROR_LIMIT) {  // 511 pattern is good.

                int errors2 = d_host_nerrors63[j];
                // we should have either field 1 (== PN63) or field 2 (== ~PN63)
                if (errors2 <= PN63_ERROR_LIMIT) {
                    GR_LOG_DEBUG(_debug_logger, "Found FIELD_SYNC_1")
                    d_field_num = 1;    // We are in field number 1 now
                    d_segment_num = -1; // This is the first segment
                } else if (errors2 >= (LENGTH_2ND_63 - PN63_ERROR_LIMIT)) {
                    GR_LOG_DEBUG(_debug_logger, "Found FIELD_SYNC_2")
                    d_field_num = 2;    // We are in field number 2 now
                    d_segment_num = -1; // This is the first segment
                } else {
                    // should be extremely rare.
                    GR_LOG_WARN(_logger,
                                std::string("PN63 error count = ") +
                                    std::to_string(errors2));
                }
            }

            if (d_field_num == 1 || d_field_num == 2) { // If we have sync
                // So we copy out current packet data to an output packet and fill its
                // plinfo

                memcpy(&out[output_produced * ATSC_DATA_SEGMENT_LENGTH],
                       &in[(i + j) * ATSC_DATA_SEGMENT_LENGTH],
                       ATSC_DATA_SEGMENT_LENGTH * sizeof(float));

                plinfo pli_out;
                pli_out.set_regular_seg((d_field_num == 2), d_segment_num);

                d_segment_num++;
                if (d_segment_num > (ATSC_SEGMENTS_PER_DATA_FIELD - 1)) {
                    d_field_num = 0;
                    d_segment_num = 0;
                } else {

                    // std::cout << flags << " " << segno << std::endl;
                    plout[output_produced++] = pli_out;
                }
            }
        }
    }

    consume_each(noutput_items,work_input);
    produce_each(output_produced,work_output);
    return work_return_code_t::WORK_OK;
}

} /* namespace dtv */
} /* namespace gr */
