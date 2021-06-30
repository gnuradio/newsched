/* -*- c++ -*- */
/*
 * Copyright 2021 Joshua Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */
//#include "multiply_const_cuda.cuh"
#include "multiply_const_cuda.hh"
#include <cusp/multiply_const.cuh>
#include <volk/volk.h>

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
    : multiply_const<T>(args), d_k(args.k), d_vlen(args.vlen)
{
    p_kernel = std::make_shared<cusp::multiply_const<T>>(d_k);
    p_kernel->cusp::multiply_const<T>::occupancy(
        &d_block_size, &d_min_grid_size
    );
    GR_LOG_INFO(gr::node::_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
    p_kernel->cusp::kernel::set_stream(d_stream);

}

template <>
multiply_const_cuda<gr_complex>::multiply_const_cuda(
    const typename multiply_const<gr_complex>::block_args& args)
    : multiply_const<gr_complex>(args), d_k(args.k), d_vlen(args.vlen)
{
    p_kernel = std::make_shared<cusp::multiply_const<std::complex<float>>>(
        std::complex<float>(real(d_k), imag(d_k))
    );
    p_kernel->cusp::multiply_const<std::complex<float>>::occupancy(
        &d_block_size, &d_min_grid_size
    );
    GR_LOG_INFO(_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
    p_kernel->cusp::kernel::set_stream(d_stream);
}

template <class T>
work_return_code_t
multiply_const_cuda<T>::work(std::vector<block_work_input>& work_input,
                             std::vector<block_work_output>& work_output)
{
    // Pre-generate these from modtool, for example
    auto in = static_cast<T*>(work_input[0].items());
    auto out = static_cast<T*>(work_output[0].items());

    auto noutput_items = work_output[0].n_items;

    int gridSize = (noutput_items + d_block_size - 1) / d_block_size;
    p_kernel->cusp::kernel::set_block_and_grid(d_block_size, gridSize);
    p_kernel->cusp::multiply_const<T>::launch(
        {in}, {out}, noutput_items
    );

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

template <>
work_return_code_t
multiply_const_cuda<gr_complex>::work(std::vector<block_work_input>& work_input,
                             std::vector<block_work_output>& work_output)
{
    // Pre-generate these from modtool, for example
    auto in = static_cast<std::complex<float>*>(work_input[0].items());
    auto out = static_cast<std::complex<float>*>(work_output[0].items());

    auto noutput_items = work_output[0].n_items;

    int gridSize = (noutput_items + d_block_size - 1) / d_block_size;
    p_kernel->cusp::kernel::set_block_and_grid(d_block_size, gridSize);

    p_kernel->cusp::multiply_const<std::complex<float>>::launch(
        {in}, {out}, noutput_items
    );

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
