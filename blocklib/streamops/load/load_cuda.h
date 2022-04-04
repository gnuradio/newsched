/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/streamops/load.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gr {
namespace streamops {

class load_cuda : public load
{
public:
    load_cuda(block_args args);
    virtual work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;

protected:
    size_t d_load;

    int d_block_size;
    int d_min_grid_size;
    cudaStream_t d_stream;
};

} // namespace streamops
} // namespace gr