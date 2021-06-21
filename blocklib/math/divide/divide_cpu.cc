/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "divide_cpu.hh"
#include <volk/volk.h>

namespace gr {
namespace math {

template <class T>
typename divide<T>::sptr divide<T>::make_cpu(const block_args& args)
{
    return std::make_shared<divide_cpu<T>>(args);
}

template <class T>
divide_cpu<T>::divide_cpu(const typename divide<T>::block_args& args)
    : divide<T>(args), d_num_inputs(args.num_inputs), d_vlen(args.vlen)
{
}

template <>
work_return_code_t
divide_cpu<float>::work(std::vector<block_work_input>& work_input,
                                std::vector<block_work_output>& work_output)
{
    auto optr = static_cast<float*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;

    auto numerator = static_cast<float *>(work_input[0].items());
    for (size_t inp = 1; inp < d_num_inputs; ++inp) {
        volk_32f_x2_divide_32f(
            optr, numerator, static_cast<float *>(work_input[inp].items()), noutput_items * d_vlen);
        numerator = optr;
    }

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}

template <>
work_return_code_t
divide_cpu<gr_complex>::work(std::vector<block_work_input>& work_input,
                                     std::vector<block_work_output>& work_output)
{
    auto optr = static_cast<gr_complex*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;

    auto numerator = static_cast<gr_complex *>(work_input[0].items());
    for (size_t inp = 1; inp < d_num_inputs; ++inp) {
        volk_32fc_x2_divide_32fc(
            optr, numerator, static_cast<gr_complex *>(work_input[inp].items()), noutput_items * d_vlen);
        numerator = optr;
    }

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}

template <class T>
work_return_code_t
divide_cpu<T>::work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output)
{
    T* optr = static_cast<T*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;

    for (size_t i = 0; i < noutput_items * d_vlen; i++) {
        T acc = (static_cast<T*>(work_input[0].items()))[i];
        for (size_t j = 1; j < d_num_inputs; j++) {
            acc /= (static_cast<T*>(work_input[j].items()))[i];
        }
        *optr++ = static_cast<T>(acc);
    }

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

template class divide<std::int16_t>;
template class divide<std::int32_t>;
template class divide<float>;
template class divide<gr_complex>;

} /* namespace math */
} /* namespace gr */
