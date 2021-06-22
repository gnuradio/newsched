/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "moving_average_cpu.hh"
#include <volk/volk.h>

namespace gr {
namespace filter {

template <class T>
typename moving_average<T>::sptr
moving_average<T>::make_cpu(const moving_average<T>::block_args& args)
{
    return std::make_shared<moving_average_cpu<T>>(args);
}

template <class T>
moving_average_cpu<T>::moving_average_cpu(
    const typename moving_average<T>::block_args& args)
    : moving_average<T>(args),
      d_length(args.length),
      d_scale(args.scale),
      d_max_iter(args.max_iter),
      d_vlen(args.vlen),
      d_new_length(args.length),
      d_new_scale(args.scale)
{
    d_sum = std::vector<T>(d_vlen);
    d_history = std::vector<T>(d_vlen * (d_length - 1));
}

template <class T>
work_return_code_t
moving_average_cpu<T>::work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output)
{
    if (d_updated) {
        d_length = d_new_length;
        d_scale = d_new_scale;
        d_updated = false;
        work_output[0].n_produced = 0;
        return work_return_code_t::WORK_OK;
    }

    auto in = static_cast<const T*>(work_input[0].items());
    auto out = static_cast<T*>(work_output[0].items());

    size_t noutput_items = std::min(work_input[0].n_items, work_output[0].n_items);

    auto num_iter = (noutput_items > d_max_iter) ? d_max_iter : noutput_items;


    for (size_t i = 0; i < num_iter; i++) {
        for (size_t elem = 0; elem < d_vlen; elem++) {
            d_sum[elem] += in[i * d_vlen + elem];
            out[i * d_vlen + elem] = d_sum[elem] * d_scale;

            if (i >= d_length - 1) {
                d_sum[elem] -= in[(i - d_length + 1) * d_vlen + elem];
            } else {
                d_sum[elem] -= d_history[i * d_vlen + elem];
            }
        }
    }

    // Stash the history (since GR 4.0 no longer has block history)
    memcpy(
        d_history.data(), &in[num_iter - d_length], d_vlen * sizeof(T) * (d_length - 1));


    work_output[0].n_produced = num_iter;
    return work_return_code_t::WORK_OK;
}

template class moving_average<int16_t>;
template class moving_average<int32_t>;
template class moving_average<float>;
template class moving_average<gr_complex>;

} /* namespace filter */
} /* namespace gr */
