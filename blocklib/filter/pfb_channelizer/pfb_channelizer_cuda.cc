/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "pfb_channelizer_cuda.hh"

#include <gnuradio/helper_cuda.h>

namespace gr {
namespace filter {

template <class T>
typename pfb_channelizer<T>::sptr
pfb_channelizer<T>::make_cuda(const pfb_channelizer<T>::block_args& args)
{
    return std::make_shared<pfb_channelizer_cuda<T>>(args);
}

template <class T>
pfb_channelizer_cuda<T>::pfb_channelizer_cuda(
    const typename pfb_channelizer<T>::block_args& args)
    : pfb_channelizer<T>(args),
      d_oversample_rate(args.oversample_rate),
      d_nfilts(args.numchans),
      d_fft(args.numchans)
{
    d_dev_taps.resize(d_nfilts);

    set_taps(args.taps);

    checkCudaErrors(cudaMalloc(d_dev_fftbuf, sizeof(gr_complex)*d_nfilts));
    d_host_fftbuf.resize(d_nfilts);

    // because we need a stream_to_streams block for the input,
    // only send tags from in[i] -> out[i].
    this->set_tag_propagation_policy(tag_propagation_policy_t::TPP_ONE_TO_ONE);
}

template <class T>
work_return_code_t pfb_channelizer_cuda<T>::set_taps(const std::vector<float>& taps)
{
    unsigned int i, j;
    unsigned int ntaps = taps.size();
    d_taps_per_filter = (unsigned int)ceil((double)ntaps / (double)d_nfilts);

    // Create d_numchan vectors to store each channel's taps
    d_taps.resize(d_nfilts);

    // Make a vector of the taps plus fill it out with 0's to fill
    // each polyphase filter with exactly d_taps_per_filter
    std::vector<float> tmp_taps = taps;
    while ((float)(tmp_taps.size()) < d_nfilts * d_taps_per_filter) {
        tmp_taps.push_back(0.0);
    }

    // Partition the filter
    for (i = 0; i < d_nfilts; i++) {
        // Each channel uses all d_taps_per_filter with 0's if not enough taps to fill out
        d_taps[i] = std::vector<float>(d_taps_per_filter, 0);
        for (j = 0; j < d_taps_per_filter; j++) {
            d_taps[i][j] = tmp_taps[i + j * d_nfilts];
        }

        // Set the filter taps for each channel
        // d_fir_filters[i].set_taps(d_taps[i]);
        std::reverse(d_taps[i].begin(), d_taps[i].end());

        // Copy to GPU memory
        if (d_dev_taps[i]) {
            checkCudaErrors(cudaFree(d_dev_taps[i]))
        }
        checkCudaErrors(cudaMalloc(d_dev_taps[i], d_taps_per_filter * sizeof(float)));
        checkCudaErrors(cudaMemcpy(
            d_dev_taps[i], d_taps[i].data(), d_taps_per_filter * sizeof(float)));
    }

    d_history = d_nfilts - 1;
}

template <class T>
work_return_code_t
pfb_channelizer_cuda<T>::work(std::vector<block_work_input>& work_input,
                              std::vector<block_work_output>& work_output)
{
    std::scoped_lock guard(d_mutex);

    const T* in = (const T*)work_input[0].items();
    T* out = (T*)work_output[0].items();
    auto noutput_items = work_output[0].n_items;

    int n = 1, i = -1, j = 0, oo = 0, last;
    int toconsume = (int)rintf(noutput_items / d_oversample_rate) - (d_history - 1);
    while (n <= toconsume) {
        j = 0;
        i = (i + d_rate_ratio) % d_nfilts;
        last = i;
        while (i >= 0) {
            in = (gr_complex*)work_input[j].items();
            d_host_fftbuf[d_idxlut[j]] = d_fir_filters[i].filter(&in[n]);
            j++;
            i--;
        }

        i = d_nfilts - 1;
        while (i > last) {
            in = (gr_complex*)work_input[j].items();
            d_host_fftbuf[d_idxlut[j]] = d_fir_filters[i].filter(&in[n - 1]);
            j++;
            i--;
        }

        n += (i + d_rate_ratio) >= (int)d_nfilts;

        // despin through FFT
        checkCudaErrors(cudaMemcpyAsync(d_dev_fft_buf, d_host_fft_buf.data(), sizeof(gr_complex)*d_nfilts, cudaMemcpyHostToDevice));
        d_fft.execute(d_dev_fft_buf, out);

        // Send to output channels
        for (unsigned int nn = 0; nn < noutputs; nn++) {
            out = (gr_complex*)work_output[nn].items();
            out[oo] = d_fft.get_outbuf()[d_channel_map[nn]];
        }
        oo++;
    }


    this->consume_each(toconsume, work_input);
    this->produce_each(noutput_items - (d_history - 1), work_output);
    return work_return_code_t::WORK_OK;
}

// template class pfb_channelizer<float>;
template class pfb_channelizer<gr_complex>;

} /* namespace filter */
} /* namespace gr */
