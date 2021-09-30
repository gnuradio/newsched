/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "pfb_channelizer_cpu.hh"
#include <volk/volk.h>

namespace gr {
namespace filter {

template <class T>
typename pfb_channelizer<T>::sptr
pfb_channelizer<T>::make_cpu(const pfb_channelizer<T>::block_args& args)
{
    return std::make_shared<pfb_channelizer_cpu<T>>(args);
}

template <class T>
pfb_channelizer_cpu<T>::pfb_channelizer_cpu(
    const typename pfb_channelizer<T>::block_args& args)
    : block("pfb_channelizer"), 
      pfb_channelizer<T>(args),

      polyphase_filterbank(args.numchans, args.taps),
      d_oversample_rate(args.oversample_rate),
      d_nchans(args.numchans)
{

    // The over sampling rate must be rationally related to the number of channels
    // in that it must be N/i for i in [1,N], which gives an outputsample rate
    // of [fs/N, fs] where fs is the input sample rate.
    // This tests the specified input sample rate to see if it conforms to this
    // requirement within a few significant figures.
    const double srate = d_nfilts / d_oversample_rate;
    const double rsrate = round(srate);
    if (fabs(srate - rsrate) > 0.00001)
        throw std::invalid_argument(
            "pfb_channelizer: oversample rate must be N/i for i in [1, N]");

    this->set_relative_rate(d_oversample_rate);

    // Default channel map. The channel map specifies which input
    // goes to which output channel; so out[0] comes from
    // channel_map[0].
    d_channel_map.resize(d_nfilts);
    for (unsigned int i = 0; i < d_nfilts; i++) {
        d_channel_map[i] = i;
    }

    // We use a look up table to set the index of the FFT input
    // buffer, which equivalently performs the FFT shift operation
    // on every other turn when the rate_ratio>1.  Also, this
    // performs the index 'flip' where the first input goes into the
    // last filter. In the pfb_decimator_ccf, we directly index the
    // input_items buffers starting with this last; here we start
    // with the first and put it into the fft object properly for
    // the same effect.
    d_rate_ratio = (int)rintf(d_nfilts / d_oversample_rate);
    d_idxlut.resize(d_nfilts);
    for (unsigned int i = 0; i < d_nfilts; i++) {
        d_idxlut[i] = d_nfilts - ((i + d_rate_ratio) % d_nfilts) - 1;
    }

    // // Calculate the number of filtering rounds to do to evenly
    // // align the input vectors with the output channels
    // d_output_multiple = 1;
    // while ((d_output_multiple * d_rate_ratio) % d_nfilts != 0)
    //     d_output_multiple++;
    // this->set_output_multiple(d_output_multiple);

    // Use set_taps to also set the history requirement
    set_taps(args.taps);

    // because we need a stream_to_streams block for the input,
    // only send tags from in[i] -> out[i].
    this->set_tag_propagation_policy(tag_propagation_policy_t::TPP_ONE_TO_ONE);
}

template <class T>
void pfb_channelizer_cpu<T>::set_taps(const std::vector<float>& taps)
{
    // std::scoped_lock guard(d_mutex);

    polyphase_filterbank::set_taps(taps);
    // set_history(d_taps_per_filter + 1);
    d_history = d_nchans * (d_taps_per_filter + 1);
    d_updated = true;
}


template <class T>
work_return_code_t
pfb_channelizer_cpu<T>::work(std::vector<block_work_input>& work_input,
                             std::vector<block_work_output>& work_output)
{
    // std::scoped_lock guard(d_mutex);

    const T* in = (const T*)work_input[0].items();
    T* out = (T*)work_output[0].items();
    auto noutput_items = work_output[0].n_items;
    auto ninput_items = work_input[0].n_items;

    if ((size_t)ninput_items < (noutput_items * d_nchans + d_history) )
    {
        return work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS;
    }

    if (d_updated) {
        d_updated = false;
        this->consume_each(0, work_input);
        this->produce_each(0, work_output);
        return work_return_code_t::WORK_OK; // history requirements may have changed.
    }

    std::vector<std::vector<T>> deinterleaved(d_nchans);

    auto total_items = std::min(ninput_items / d_nchans, (size_t)noutput_items);

    for (size_t j = 0; j < d_nchans; j++) {
        if (deinterleaved[j].size() < total_items) {
            deinterleaved[j].resize(total_items);
        }
        for (size_t i = 0; i < total_items; i++) {
            deinterleaved[j][i] = in[i * d_nchans + j];
        }
    }

    size_t noutputs = work_output.size();

    // The following algorithm looks more complex in order to handle
    // the cases where we want more that 1 sps for each
    // channel. Otherwise, this would boil down into a single loop
    // that operates from input_items[0] to [d_nfilts].

    // When dealing with osps>1, we start not at the last filter,
    // but nfilts/osps and then wrap around to the next symbol into
    // the other set of filters.
    // For details of this operation, see:
    // fred harris, Multirate Signal Processing For Communication
    // Systems. Upper Saddle River, NJ: Prentice Hall, 2004.

    int n = 1, i = -1, j = 0, oo = 0, last;
    int toconsume = (int)rintf((noutput_items * d_nchans) / d_oversample_rate) - (d_history - 1);
    while (n <= toconsume) {
        j = 0;
        i = (i + d_rate_ratio) % d_nfilts;
        last = i;
        while (i >= 0) {
            // in = (gr_complex*)work_input[j].items();
            d_fft.get_inbuf()[d_idxlut[j]] =
                d_fir_filters[i].filter(&deinterleaved[j][n]);
            j++;
            i--;
        }

        i = d_nfilts - 1;
        while (i > last) {
            // in = (gr_complex*)work_input[j].items();
            d_fft.get_inbuf()[d_idxlut[j]] =
                d_fir_filters[i].filter(&deinterleaved[j][n - 1]);
            j++;
            i--;
        }

        n += (i + d_rate_ratio) >= (int)d_nfilts;

        // despin through FFT
        d_fft.execute();

        // Send to output channels
        for (unsigned int nn = 0; nn < noutputs; nn++) {
            out = (gr_complex*)work_output[nn].items();
            out[oo] = d_fft.get_outbuf()[d_channel_map[nn]];
        }
        oo++;
    }

    this->consume_each(toconsume, work_input);
    // this->produce_each(noutput_items - (d_history / d_nchans - 1), work_output);
    this->produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}

// template class pfb_channelizer<float>;
template class pfb_channelizer<gr_complex>;

} /* namespace filter */
} /* namespace gr */
