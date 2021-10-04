/* -*- c++ -*- */
/*
 * Copyright 2021 Joshua Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "multiply_const_cuda.hh"
#include <volk/volk.h>
#include <thrust/complex.h>

namespace gr {
namespace math {

template <class T>
typename multiply_const<T>::sptr multiply_const<T>::make_cuda(const block_args& args)
{
    return std::make_shared<multiply_const_cuda<T>>(args);
}

template <class T>
multiply_const_cuda<T>::multiply_const_cuda(
    const typename multiply_const<T>::block_args& args)
    : sync_block("multiply_const_cuda"), multiply_const<T>(args), d_k(args.k), d_vlen(args.vlen)
{
    p_kernel = std::make_shared<cusp::multiply_const<T>>(args.k);

    cudaStreamCreate(&d_stream);
    p_kernel->set_stream(d_stream);
}

template <>
multiply_const_cuda<gr_complex>::multiply_const_cuda(
    const typename multiply_const<gr_complex>::block_args& args)
    : sync_block("multiply_const_cuda"), multiply_const<gr_complex>(args), d_k(args.k), d_vlen(args.vlen)
{
    p_kernel = std::make_shared<cusp::multiply_const<gr_complex>>((thrust::complex<float>)args.k);

    cudaStreamCreate(&d_stream);
    p_kernel->set_stream(d_stream);
}

template <class T>
work_return_code_t
multiply_const_cuda<T>::work(std::vector<block_work_input>& work_input,
                             std::vector<block_work_output>& work_output)
{
    auto noutput_items = work_output[0].n_items;

    p_kernel->launch_default_occupancy({work_input[0].items()}, { work_output[0].items() }, noutput_items);
    cudaStreamSynchronize(d_stream);

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

template class multiply_const<std::int16_t>;
template class multiply_const<std::int32_t>;
template class multiply_const<float>;
template class multiply_const<gr_complex>;

} /* namespace math */
} /* namespace gr */
