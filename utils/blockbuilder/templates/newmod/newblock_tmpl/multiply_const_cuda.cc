/* -*- c++ -*- */
/*
 * Copyright 2021 Joshua Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */
#include "multiply_const_cuda.cuh"
#include "multiply_const_cuda.hh"
#include <volk/volk.h>

namespace gr {
namespace blocks {

template <class T>
typename multiply_const<T>::sptr multiply_const<T>::make_cuda(const block_args& args)
{
    return std::make_shared<multiply_const_cuda<T>>(args);
}

template <class T>
multiply_const_cuda<T>::multiply_const_cuda(
    const typename multiply_const<T>::block_args& args)
    : multiply_const<T>(args), d_k(args.k), d_vlen(args.vlen)
{
    multiply_const_cu::get_block_and_grid<T>(&d_min_grid_size, &d_block_size);
    GR_LOG_INFO(gr::node::_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
}

template <>
multiply_const_cuda<gr_complex>::multiply_const_cuda(
    const typename multiply_const<gr_complex>::block_args& args)
    : multiply_const<gr_complex>(args), d_k(args.k), d_vlen(args.vlen)
{
    multiply_const_cu::get_block_and_grid<cuFloatComplex>(&d_min_grid_size, &d_block_size);
    GR_LOG_INFO(_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
}


template <class T>
work_return_code_t
multiply_const_cuda<T>::work(std::vector<block_work_input>& work_input,
                             std::vector<block_work_output>& work_output)
{
    auto in = work_input[0].items<T>());
    auto out = work_output[0].items<T>());

    auto noutput_items = work_output[0].n_items;

    multiply_const_cu::exec_kernel(
        in, out, d_k, (noutput_items * d_vlen) / d_block_size, d_block_size, d_stream);

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

template <>
work_return_code_t
multiply_const_cuda<gr_complex>::work(std::vector<block_work_input>& work_input,
                             std::vector<block_work_output>& work_output)
{
    auto in = work_input[0].items<cuFloatComplex>());
    auto out = work_output[0].items<cuFloatComplex>());

    auto noutput_items = work_output[0].n_items;

    auto k_cufc = make_cuFloatComplex(real(d_k), imag(d_k));

    multiply_const_cu::exec_kernel(
        in, out, k_cufc, (noutput_items * d_vlen) / d_block_size, d_block_size, d_stream);

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
