/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "iir_filter_cpu.h"
#include "iir_filter_cpu_gen.h"

namespace gr {
namespace filter {

template <class T_IN, class T_OUT, class TAP_T>
iir_filter_cpu<T_IN, T_OUT, TAP_T>::iir_filter_cpu(
    const typename iir_filter<T_IN, T_OUT, TAP_T>::block_args& args)
    : INHERITED_CONSTRUCTORS(T_IN, T_OUT, TAP_T),
      d_iir(args.fftaps, args.fbtaps, args.oldstyle)
{
}

template <class T_IN, class T_OUT, class TAP_T>
work_return_code_t
iir_filter_cpu<T_IN, T_OUT, TAP_T>::work(std::vector<block_work_input_sptr>& work_input,
                                         std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0]->items<T_IN>();
    auto out = work_output[0]->items<T_OUT>();
    auto noutput_items = work_output[0]->n_items;

    d_iir.filter_n(out, in, noutput_items);
    this->produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}

} /* namespace filter */
} /* namespace gr */
