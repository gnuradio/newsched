/* -*- c++ -*- */
/*
 * Copyright 2021 Joshua Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "conjugate_cuda.hh"
#include <cusp/conjugate.cuh>
#include <volk/volk.h>

namespace gr {
namespace math {

conjugate::sptr conjugate::make_cuda(const block_args& args)
{
    return std::make_shared<conjugate_cuda>(args);
}

conjugate_cuda::conjugate_cuda(
    const typename conjugate::block_args& args)
    : conjugate(args)
{
    p_kernel = std::make_shared<cusp::conjugate>();
    p_kernel->cusp::conjugate::occupancy(
        &d_block_size, &d_min_grid_size
    );
    GR_LOG_INFO(gr::node::_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
    p_kernel->cusp::kernel::set_stream(d_stream);

}

work_return_code_t
conjugate_cuda::work(std::vector<block_work_input>& work_input,
                             std::vector<block_work_output>& work_output)
{
    // Pre-generate these from modtool, for example
    auto in = static_cast<std::complex<float>*>(work_input[0].items());
    auto out = static_cast<std::complex<float>*>(work_output[0].items());

    auto noutput_items = work_output[0].n_items;

    int gridSize = (noutput_items + d_block_size - 1) / d_block_size;
    p_kernel->cusp::kernel::set_block_and_grid(d_block_size, gridSize);
    p_kernel->cusp::conjugate::launch(
        {in}, {out}, noutput_items
    );

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

} /* namespace blocks */
} /* namespace gr */
