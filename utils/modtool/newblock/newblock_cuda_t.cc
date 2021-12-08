/* -*- c++ -*- */
/*
 * Copyright 2021 Joshua Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */
#include "newblock_cuda.cuh"
#include "newblock_cuda.hh"
#include "newblock_cuda_gen.hh"
#include <volk/volk.h>

namespace gr {
namespace newmod {

template <class T>
newblock_cuda<T>::newblock_cuda(
    const typename newblock<T>::block_args& args)
    : newblock<T>(args), d_k(args.k), d_vlen(args.vlen)
{
    newblock_cu::get_block_and_grid<T>(&d_min_grid_size, &d_block_size);
    GR_LOG_INFO(gr::node::_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
}

template <>
newblock_cuda<gr_complex>::newblock_cuda(
    const typename newblock<gr_complex>::block_args& args)
    : newblock<gr_complex>(args), d_k(args.k), d_vlen(args.vlen)
{
    newblock_cu::get_block_and_grid<cuFloatComplex>(&d_min_grid_size, &d_block_size);
    GR_LOG_INFO(_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
}


template <class T>
work_return_code_t
newblock_cuda<T>::work(std::vector<block_work_input_sptr>& work_input,
                             std::vector<block_work_output_sptr>& work_output)
{
    // Do block specific code here
    return work_return_code_t::WORK_OK;
}

template <>
work_return_code_t
newblock_cuda<gr_complex>::work(std::vector<block_work_input_sptr>& work_input,
                             std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0].items<cuFloatComplex>());
    auto out = work_output[0].items<cuFloatComplex>());

    auto noutput_items = work_output[0].n_items;

    auto k_cufc = make_cuFloatComplex(real(d_k), imag(d_k));

    newblock_cu::exec_kernel(
        in, out, k_cufc, (noutput_items * d_vlen) / d_block_size, d_block_size, d_stream);

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

template class newblock<std::int16_t>;
template class newblock<std::int32_t>;
template class newblock<float>;
template class newblock<gr_complex>;

} /* namespace newmod */
} /* namespace gr */
