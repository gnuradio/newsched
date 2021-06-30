/* -*- c++ -*- */
/*
 * Copyright 2021 Joshua Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */
#include "multiply_cuda.hh"
#include <cusp/multiply.cuh>
#include <volk/volk.h>

namespace gr {
namespace math {

template <class T>
typename multiply<T>::sptr multiply<T>::make_cuda(const block_args& args)
{
    return std::make_shared<multiply_cuda<T>>(args);
}

template <class T>
multiply_cuda<T>::multiply_cuda(
    const typename multiply<T>::block_args& args)
    : multiply<T>(args), num_inputs(args.num_inputs), d_vlen(args.vlen)
{
    p_kernel = std::make_shared<cusp::multiply<T>>(num_inputs);
    p_kernel->cusp::multiply<T>::occupancy(
        &d_block_size, &d_min_grid_size
    );
    GR_LOG_INFO(gr::node::_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
    p_kernel->cusp::kernel::set_stream(d_stream);

}

template <>
multiply_cuda<gr_complex>::multiply_cuda(
    const typename multiply<gr_complex>::block_args& args)
    : multiply<gr_complex>(args), num_inputs(args.num_inputs), d_vlen(args.vlen)
{
    p_kernel = std::make_shared<cusp::multiply<std::complex<float>>>(num_inputs);
    p_kernel->cusp::multiply<std::complex<float>>::occupancy(
        &d_block_size, &d_min_grid_size
    );
    GR_LOG_INFO(_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
    p_kernel->cusp::kernel::set_stream(d_stream);
}

template <class T>
work_return_code_t multiply_cuda<T>::work(std::vector<block_work_input>& work_input,
                                          std::vector<block_work_output>& work_output)
{
    auto out = static_cast<T*>(work_output[0].items());
    std::vector<const void*> input_data_pointer_vec(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
        input_data_pointer_vec[i] = work_input[i].items();
    }

    auto noutput_items = work_output[0].n_items;

    int gridSize = (noutput_items + d_block_size - 1) / d_block_size;
    p_kernel->cusp::kernel::set_block_and_grid(d_block_size, gridSize);
    p_kernel->cusp::multiply<T>::launch(input_data_pointer_vec, { out }, noutput_items);

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

template <>
work_return_code_t
multiply_cuda<gr_complex>::work(std::vector<block_work_input>& work_input,
                                std::vector<block_work_output>& work_output)
{
    auto out = static_cast<std::complex<float>*>(work_output[0].items());

    std::vector<const void*> input_data_pointer_vec(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
        input_data_pointer_vec[i] = work_input[i].items();
    }

    auto noutput_items = work_output[0].n_items;

    int gridSize = (noutput_items + d_block_size - 1) / d_block_size;
    p_kernel->cusp::kernel::set_block_and_grid(d_block_size, gridSize);

    p_kernel->cusp::multiply<std::complex<float>>::launch(
        input_data_pointer_vec, { out }, noutput_items);

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

template class multiply<std::int16_t>;
template class multiply<std::int32_t>;
template class multiply<float>;
template class multiply<gr_complex>;

} /* namespace blocks */
} /* namespace gr */
