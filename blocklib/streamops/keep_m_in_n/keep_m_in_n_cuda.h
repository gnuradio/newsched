/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/streamops/keep_m_in_n.h>

#include <cusp/keep_m_in_n.cuh>

namespace gr {
namespace streamops {

class keep_m_in_n_cuda : public keep_m_in_n
{
public:
    keep_m_in_n_cuda(block_args args);
    virtual work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;

private:
    int d_min_block;
    int d_min_grid;
    cudaStream_t d_stream;
    std::shared_ptr<cusp::keep_m_in_n<uint8_t>> p_kernel;
    void on_parameter_change(param_action_sptr action) override;
};

} // namespace streamops
} // namespace gr