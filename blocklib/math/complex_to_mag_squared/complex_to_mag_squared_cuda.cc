/* -*- c++ -*- */
/*
 * Copyright 2021 Joshua Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "complex_to_mag_squared_cuda.hh"
#include <cusp/complex_to_mag_squared.cuh>
#include <volk/volk.h>

namespace gr {
namespace math {

complex_to_mag_squared::sptr complex_to_mag_squared::make_cuda(const block_args& args)
{
    return std::make_shared<complex_to_mag_squared_cuda>(args);
}

complex_to_mag_squared_cuda::complex_to_mag_squared_cuda(
    const typename complex_to_mag_squared::block_args& args)
    : complex_to_mag_squared(args), d_vlen(args.vlen)
{
    p_kernel = std::make_shared<cusp::complex_to_mag_squared>();
    p_kernel->cusp::complex_to_mag_squared::occupancy(
        &d_block_size, &d_min_grid_size
    );
    GR_LOG_INFO(gr::node::_logger, "minGrid: {}, blockSize: {}", d_min_grid_size, d_block_size);
    cudaStreamCreate(&d_stream);
    p_kernel->cusp::kernel::set_stream(d_stream);

}

work_return_code_t
complex_to_mag_squared_cuda::work(std::vector<block_work_input>& work_input,
                          std::vector<block_work_output>& work_output)
{
    // Pre-generate these from modtool, for example
    auto in = static_cast<std::complex<float>*>(work_input[0].items());
    auto out = static_cast<std::complex<float>*>(work_output[0].items());

    auto noutput_items = work_output[0].n_items;

    int gridSize = (noutput_items + d_block_size - 1) / d_block_size;
    p_kernel->cusp::kernel::set_block_and_grid(d_block_size, gridSize);
    p_kernel->cusp::complex_to_mag_squared::launch(
        {in}, {out}, noutput_items
    );

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

} /* namespace blocks */
} /* namespace gr */
