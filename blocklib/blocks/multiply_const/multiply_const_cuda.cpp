/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "multiply_const_cuda.hpp"
#include <volk/volk.h>

namespace gr {
namespace blocks {

template <typename T>
void exec_multiply_const_kernel(
    const T* in, T* out, T k, int grid_size, int block_size, cudaStream_t stream);

template <typename T>
extern void get_block_and_grid_multiply_const(int* minGrid, int* minBlock);

template <class T>
typename multiply_const<T>::sptr multiply_const<T>::make_cpu(const block_args& args)
{
    return std::make_shared<multiply_const_cuda<T>>(args);
}

template <class T>
multiply_const_cuda<T>::multiply_const_cuda(
    const typename multiply_const<T>::block_args& args)
    : multiply_const<T>(args), d_k(args.k), d_vlen(args.vlen)
{
    get_block_and_grid_multiply_const<T>(&d_min_grid_size, &d_block_size);
    std::cout << "minGrid: " << d_min_grid_size << ", blockSize: " << d_block_size
              << std::endl;

    cudaStreamCreate(&d_stream);
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

    exec_multiply_const_kernel(
        in, out, d_k, (noutput_items * d_vlen) / d_block_size, d_block_size, d_stream);

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

// template class multiply_const<std::int16_t>;
// template class multiply_const<std::int32_t>;
template class multiply_const<float>;
// template class multiply_const<gr_complex>;

} /* namespace blocks */
} /* namespace gr */
