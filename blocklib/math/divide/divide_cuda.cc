/* -*- c++ -*- */
/*
 * Copyright 2021 Joshua Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */
#include "divide_cuda.hh"
#include <cusp/divide.cuh>
#include <volk/volk.h>

namespace gr {
namespace math {

template <class T>
typename divide<T>::sptr divide<T>::make_cuda(const block_args& args)
{
    return std::make_shared<divide_cuda<T>>(args);
}

template <class T>
divide_cuda<T>::divide_cuda(
    const typename divide<T>::block_args& args)
    : divide<T>(args), num_inputs(args.num_inputs), d_vlen(args.vlen)
{
    p_kernel = std::make_shared<cusp::divide<T>>(num_inputs);
    p_kernel->cusp::divide<T>::occupancy(
        &d_block_size, &d_min_grid_size
    );
    GR_LOG_INFO(gr::node::_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
    p_kernel->cusp::kernel::set_stream(d_stream);

}

template <>
divide_cuda<gr_complex>::divide_cuda(
    const typename divide<gr_complex>::block_args& args)
    : divide<gr_complex>(args), num_inputs(args.num_inputs), d_vlen(args.vlen)
{
    p_kernel = std::make_shared<cusp::divide<std::complex<float>>>(num_inputs);
    p_kernel->cusp::divide<std::complex<float>>::occupancy(
        &d_block_size, &d_min_grid_size
    );
    GR_LOG_INFO(_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
    p_kernel->cusp::kernel::set_stream(d_stream);
}

template <class T>
work_return_code_t divide_cuda<T>::work(std::vector<block_work_input>& work_input,
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
    p_kernel->cusp::divide<T>::launch(input_data_pointer_vec, { out }, noutput_items);

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

template <>
work_return_code_t
divide_cuda<gr_complex>::work(std::vector<block_work_input>& work_input,
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

    p_kernel->cusp::divide<std::complex<float>>::launch(
        input_data_pointer_vec, { out }, noutput_items);

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

template class divide<std::int16_t>;
template class divide<std::int32_t>;
template class divide<float>;
template class divide<gr_complex>;

} /* namespace blocks */
} /* namespace gr */
