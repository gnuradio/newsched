/* -*- c++ -*- */
/*
 * Copyright 2021 gr-dtv author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <gnuradio/dtv/cuda/atsc_sync_cuda.hpp>
#include <gnuradio/filter/interpolator_taps.hpp>

extern void exec_atsc_sync_and_integrate(const float* in,
                                         float* interp,
                                         float* interp_taps,
                                         int8_t* integrator_accum,
                                         float* params,
                                         cudaStream_t stream);

namespace gr {
namespace dtv {

static const double LOOP_FILTER_TAP = 0.0005; // 0.0005 works
static const double ADJUSTMENT_GAIN = 1.0e-5 / (10 * ATSC_DATA_SEGMENT_LENGTH);
static const int SYMBOL_INDEX_OFFSET = 3;
static const int MIN_SEG_LOCK_CORRELATION_VALUE = 5;
static const signed char SSI_MIN = -16;
static const signed char SSI_MAX = 15;


atsc_sync_cuda::sptr atsc_sync_cuda::make(float rate)
{
    return std::make_shared<atsc_sync_cuda>(rate);
}

atsc_sync_cuda::atsc_sync_cuda(float rate)
    : gr::block("dtv_atsc_sync_cuda"),
      d_rx_clock_to_symbol_freq(rate / ATSC_SYMBOL_RATE),
      d_si(0)
{
    add_port(port<float>::make("in", port_direction_t::INPUT));
    add_port(
        port<float>::make("out", port_direction_t::OUTPUT, { ATSC_DATA_SEGMENT_LENGTH }));

    d_loop.set_taps(LOOP_FILTER_TAP);
    // Create device memory for input and output samples

    checkCudaErrors(cudaMalloc(
        (void**)&d_dev_in,
        1500 + OUTPUT_MULTIPLE * sizeof(float) *
                   (int)(ATSC_DATA_SEGMENT_LENGTH * d_rx_clock_to_symbol_freq)));

    checkCudaErrors(cudaMallocHost(
        (void**)&d_host_in,
        1500 + OUTPUT_MULTIPLE * sizeof(float) *
                   (int)(ATSC_DATA_SEGMENT_LENGTH * d_rx_clock_to_symbol_freq)));

    checkCudaErrors(cudaMalloc(
        (void**)&d_dev_out, OUTPUT_MULTIPLE * sizeof(float) * ATSC_DATA_SEGMENT_LENGTH));


    checkCudaErrors(cudaMalloc((void**)&d_dev_integrator_accum,
                               sizeof(int8_t) * ATSC_DATA_SEGMENT_LENGTH));

    checkCudaErrors(cudaMalloc((void**)&d_dev_params, 6 * sizeof(float)));
    checkCudaErrors(cudaMallocHost((void**)&d_host_params, 6 * sizeof(float)));

    checkCudaErrors(cudaMalloc(
        (void**)&d_data_mem, OUTPUT_MULTIPLE * ATSC_DATA_SEGMENT_LENGTH * sizeof(float)));

    // Copy the interpolation filter taps into device memory
    checkCudaErrors(
        cudaMalloc((void**)&d_dev_taps, sizeof(float) * (NSTEPS + 1) * NTAPS));


    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Taps need to be reversed into device memory
    const float* ptaps = &taps[0][0];
    std::vector<float> rev_taps(ptaps, ptaps + (NSTEPS + 1) * NTAPS);
    for (int i = 0; i < (NSTEPS + 1); i++) {
        // reverse each filter
        std::reverse(rev_taps.begin() + NTAPS * i, rev_taps.begin() + NTAPS * (i + 1));
    }

    checkCudaErrors(cudaMemcpy(d_dev_taps,
                               rev_taps.data(),
                               sizeof(float) * (NSTEPS + 1) * NTAPS,
                               cudaMemcpyHostToDevice));

    reset();

    set_relative_rate(1.0 / ((int)(ATSC_DATA_SEGMENT_LENGTH * d_w) + 1));
    set_output_multiple(OUTPUT_MULTIPLE);
}

void atsc_sync_cuda::reset()
{
    d_w = d_rx_clock_to_symbol_freq;
    d_mu = 0.5;

    d_timing_adjust = 0;
    d_counter = 0;
    d_symbol_index = 0;
    d_seg_locked = false;

    d_sr = 0;
    
    cudaMemset(d_data_mem,
    // memset(d_data_mem,
               0,
               OUTPUT_MULTIPLE * ATSC_DATA_SEGMENT_LENGTH *
                   sizeof(*d_data_mem)); // (float)0 = 0x00000000

    checkCudaErrors(
        cudaMemset(d_dev_integrator_accum, SSI_MIN, ATSC_DATA_SEGMENT_LENGTH));
}

work_return_code_t atsc_sync_cuda::work(std::vector<block_work_input>& work_input,
                                        std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const float*>(work_input[0].items());
    auto out = static_cast<float*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;
    auto ninput_items = work_input[0].n_items;

    float interp_sample;

    int inputs_consumed = 0;
    int outputs_produced = 0;

    int input_mem_items = ((int)(ATSC_DATA_SEGMENT_LENGTH * d_w) + 1);
    // amount actually consumed
    d_si = 0;


    // Because this is a general block, we must do some forecasting
    auto min_items = static_cast<int>(noutput_items * d_rx_clock_to_symbol_freq *
                                      ATSC_DATA_SEGMENT_LENGTH) +
                     1500 - 1;
    if (work_input[0].n_items < min_items) {
        // consume_each(0,work_input);
        return work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS;
    }
    assert(work_output[0].n_items % OUTPUT_MULTIPLE == 0);
    // assert(noutput_items <= OUTPUT_MULTIPLE);

    if (work_output[0].nitems_written() >= 1312) {
        volatile int x = 7;
    }

    // noutput items are in vectors of ATSC_DATA_SEGMENT_LENGTH
    int no = 0;
    for (int n = 0; n < noutput_items; n += OUTPUT_MULTIPLE) {
        int d_si_start = d_si;
        double d_mu_start = d_mu;

        if ((d_si + (int)d_interp.ntaps()) >= ninput_items) {
            d_si = d_si_start;
            d_mu = d_mu_start;
            break;
        }

        // Launch 832 threads to do interpolation
        // The kernel will do 8 tap dot products, so total threads / 8

        d_host_params[0] = d_timing_adjust;
        d_host_params[1] = d_mu;
        d_host_params[2] = d_w;
        d_host_params[3] = 0;

        checkCudaErrors(cudaMemcpyAsync(d_dev_params,
                                        d_host_params,
                                        sizeof(float) * 6,
                                        cudaMemcpyHostToDevice,
                                        streams[0]));

        cudaStreamSynchronize(streams[0]);

        for (int oo = 0; oo < OUTPUT_MULTIPLE; oo++) {
            exec_atsc_sync_and_integrate(in + d_si_start,
                                         d_dev_out + oo * ATSC_DATA_SEGMENT_LENGTH,
                                         d_dev_taps,
                                         d_dev_integrator_accum,
                                         d_dev_params,
                                         streams[0]);
        }
        checkCudaErrors(cudaPeekAtLastError());
        cudaStreamSynchronize(streams[0]);

        checkCudaErrors(cudaMemcpyAsync(d_host_params,
                                        d_dev_params,
                                        sizeof(float) * 6,
                                        cudaMemcpyDeviceToHost,
                                        streams[0]));


        cudaStreamSynchronize(streams[0]);
        d_si += (int)rint(d_host_params[3]);
        d_mu = d_host_params[1];
        d_timing_adjust = d_host_params[0];

        uint16_t tmp_idx = (uint16_t)d_host_params[4];
        int16_t tmp_val = (int16_t)d_host_params[5];

        d_seg_locked = tmp_val >= MIN_SEG_LOCK_CORRELATION_VALUE;


        d_cntr += OUTPUT_MULTIPLE;

        if (d_seg_locked) {

            // int idx_start = SYMBOL_INDEX_OFFSET + tmp_idx;
            int idx_start = tmp_idx - SYMBOL_INDEX_OFFSET;
            if (idx_start >= ATSC_DATA_SEGMENT_LENGTH)
                idx_start -= ATSC_DATA_SEGMENT_LENGTH;
            if (idx_start < 0)
                idx_start += ATSC_DATA_SEGMENT_LENGTH;

            // Obviously there is a double copy here that needs to be optimized
            checkCudaErrors(cudaMemcpyAsync(
                d_data_mem + (ATSC_DATA_SEGMENT_LENGTH - idx_start),
                d_dev_out,
                sizeof(float) *
                    ((OUTPUT_MULTIPLE - 1) * ATSC_DATA_SEGMENT_LENGTH + idx_start),
                cudaMemcpyDeviceToDevice,
                streams[0]));
            cudaStreamSynchronize(streams[0]);

            checkCudaErrors(cudaMemcpyAsync(&out[no * ATSC_DATA_SEGMENT_LENGTH],
                                            d_data_mem,
                                            sizeof(float) * (ATSC_DATA_SEGMENT_LENGTH) * OUTPUT_MULTIPLE,
                                            cudaMemcpyDeviceToDevice,
                                            streams[0]));
            cudaStreamSynchronize(streams[0]);

            checkCudaErrors(cudaMemcpyAsync(
                d_data_mem,
                d_dev_out + ATSC_DATA_SEGMENT_LENGTH * (OUTPUT_MULTIPLE - 1) + idx_start,
                sizeof(float) * (ATSC_DATA_SEGMENT_LENGTH - idx_start),
                cudaMemcpyDeviceToDevice,
                streams[0]));
            cudaStreamSynchronize(streams[0]);

            no += OUTPUT_MULTIPLE;
        }
    }

    consume_each(d_si, work_input);
    produce_each(no, work_output);
    return work_return_code_t::WORK_OK;
}

} /* namespace dtv */
} /* namespace gr */

