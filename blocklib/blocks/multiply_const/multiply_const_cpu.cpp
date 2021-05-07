/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "multiply_const_cpu.hpp"
#include <volk/volk.h>

namespace gr {
namespace blocks {

template <class T>
typename multiply_const<T>::sptr multiply_const<T>::make_cpu(const block_args& args)
{
    return std::make_shared<multiply_const_cpu<T>>(args);
}

template <class T>
multiply_const_cpu<T>::multiply_const_cpu(const typename multiply_const<T>::block_args& args)
    : multiply_const<T>(args), d_k(args.k), d_vlen(args.vlen)
{
}

template <>
work_return_code_t
multiply_const_cpu<float>::work(std::vector<block_work_input>& work_input,
                                std::vector<block_work_output>& work_output)
{

    const float* in = (const float*)work_input[0].items();
    float* out = (float*)work_output[0].items();
    int noi = work_output[0].n_items * d_vlen;

    volk_32f_s32f_multiply_32f(out, in, d_k, noi);

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}

template <>
work_return_code_t
multiply_const_cpu<gr_complex>::work(std::vector<block_work_input>& work_input,
                                     std::vector<block_work_output>& work_output)
{
    const gr_complex* in = (const gr_complex*)work_input[0].items();
    gr_complex* out = (gr_complex*)work_output[0].items();
    int noi = work_output[0].n_items * d_vlen;

    volk_32fc_s32fc_multiply_32fc(out, in, d_k, noi);

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}

template <class T>
work_return_code_t
multiply_const_cpu<T>::work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output)
{
    // Pre-generate these from modtool, for example
    T* iptr = (T*)work_input[0].items();
    T* optr = (T*)work_output[0].items();

    int size = work_output[0].n_items * d_vlen;

    while (size >= 8) {
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        size -= 8;
    }

    while (size-- > 0)
        *optr++ = *iptr++ * d_k;

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

template class multiply_const<std::int16_t>;
template class multiply_const<std::int32_t>;
template class multiply_const<float>;
template class multiply_const<gr_complex>;

} /* namespace blocks */
} /* namespace gr */
