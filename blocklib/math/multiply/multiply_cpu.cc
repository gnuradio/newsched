/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "multiply_cpu.hh"
#include <volk/volk.h>

namespace gr {
namespace math {

template <class T>
typename multiply<T>::sptr multiply<T>::make_cpu(const block_args& args)
{
    return std::make_shared<multiply_cpu<T>>(args);
}

template <class T>
multiply_cpu<T>::multiply_cpu(const typename multiply<T>::block_args& args)
    : sync_block("multiply"), multiply<T>(args), d_num_inputs(args.num_inputs), d_vlen(args.vlen)
{
}

template <>
multiply_cpu<float>::multiply_cpu(const typename multiply<float>::block_args& args)
    : sync_block("multiply_ff"), multiply<float>(args), d_num_inputs(args.num_inputs), d_vlen(args.vlen)
{
    // const int alignment_multiple = volk_get_alignment() / sizeof(float);
    // set_output_multiple(std::max(1, alignment_multiple));
}

template <>
multiply_cpu<gr_complex>::multiply_cpu(const typename multiply<gr_complex>::block_args& args)
    : sync_block("multiply_cc"), multiply<gr_complex>(args), d_num_inputs(args.num_inputs), d_vlen(args.vlen)
{
    // const int alignment_multiple = volk_get_alignment() / sizeof(gr_complex);
    // set_output_multiple(std::max(1, alignment_multiple));
}

template <>
work_return_code_t
multiply_cpu<float>::work(std::vector<block_work_input>& work_input,
                                std::vector<block_work_output>& work_output)
{
    auto out = static_cast<float*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;
    int noi = d_vlen * noutput_items;

    memcpy(out, work_input[0].items(), noi * sizeof(float));
    for (size_t i = 1; i < d_num_inputs; i++) {
        volk_32f_x2_multiply_32f(out, out, static_cast<float*>(work_input[i].items()), noi);
    }

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}

template <>
work_return_code_t
multiply_cpu<gr_complex>::work(std::vector<block_work_input>& work_input,
                                     std::vector<block_work_output>& work_output)
{
    auto out = static_cast<gr_complex*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;
    int noi = d_vlen * noutput_items;

    memcpy(out, work_input[0].items(), noi * sizeof(gr_complex));
    for (size_t i = 1; i < d_num_inputs; i++) {
        volk_32fc_x2_multiply_32fc(out, out, static_cast<gr_complex*>(work_input[i].items()), noi);
    }

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}

template <class T>
work_return_code_t
multiply_cpu<T>::work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output)
{
    T* optr = static_cast<T*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;

    for (size_t i = 0; i < noutput_items * d_vlen; i++) {
        T acc = (static_cast<T*>(work_input[0].items()))[i];
        for (size_t j = 1; j < d_num_inputs; j++) {
            acc *= (static_cast<T*>(work_input[j].items()))[i];
        }
        *optr++ = static_cast<T>(acc);
    }

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

template class multiply<std::int16_t>;
template class multiply<std::int32_t>;
template class multiply<float>;
template class multiply<gr_complex>;

} /* namespace math */
} /* namespace gr */
