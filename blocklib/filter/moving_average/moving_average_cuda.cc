/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "moving_average_cuda.hh"

namespace gr {
namespace filter {

template <class T>
typename moving_average<T>::sptr
moving_average<T>::make_cuda(const moving_average<T>::block_args& args)
{
    return std::make_shared<moving_average_cuda<T>>(args);
}

template <class T>
moving_average_cuda<T>::moving_average_cuda(
    const typename moving_average<T>::block_args& args)
    : block("moving_average_cuda"),
      moving_average<T>(args),
      d_length(args.length),
      d_scale(args.scale),
      d_max_iter(args.max_iter),
      d_vlen(args.vlen),
      d_new_length(args.length),
      d_new_scale(args.scale)
{
    std::vector<T> taps(d_length);
    for (size_t i = 0; i < d_length; i++) {
        taps[i] = (float)1.0 * d_scale;
    }

    p_kernel_full =
        std::make_shared<cusp::convolve<T, T>>(taps, cusp::convolve_mode_t::FULL_TRUNC);
    p_kernel_valid =
        std::make_shared<cusp::convolve<T, T>>(taps, cusp::convolve_mode_t::VALID);
}

template <class T>
work_return_code_t
moving_average_cuda<T>::work(std::vector<block_work_input>& work_input,
                             std::vector<block_work_output>& work_output)
{
    if (work_input[0].n_items < (int)d_length) {
        work_output[0].n_produced = 0;
        work_input[0].n_consumed = 0;
        return work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS;
    }

    auto in = work_input[0].items<T>();
    auto out = work_output[0].items<T>();

    size_t noutput_items =
        std::min((int)(work_input[0].n_items), work_output[0].n_items);

    // auto num_iter = (noutput_items > d_max_iter) ? d_max_iter : noutput_items;
    auto num_iter = noutput_items;
    auto tr = work_input[0].buffer->total_read();

    if (tr == 0) {
        p_kernel_full->launch_default_occupancy({ in }, { out }, num_iter);
    } else {
        p_kernel_valid->launch_default_occupancy({ in }, { out }, num_iter);
    }

    // don't consume the last d_length-1 samples
    work_output[0].n_produced = tr == 0 ? num_iter : num_iter - (d_length - 1);
    work_input[0].n_consumed =
        tr == 0 ? num_iter - (d_length - 1) : num_iter - (d_length - 1);
    return work_return_code_t::WORK_OK;
} // namespace filter

// template class moving_average<int16_t>;
// template class moving_average<int32_t>;
template class moving_average<float>;
template class moving_average<gr_complex>;

} // namespace filter
} /* namespace gr */
