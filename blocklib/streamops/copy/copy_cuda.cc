/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "copy_cuda.h"
#include "copy_cuda_gen.h"

#include <gnuradio/helper_cuda.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>


namespace gr {
namespace streamops {

copy_cuda::copy_cuda(block_args args) : INHERITED_CONSTRUCTORS
{
    cudaStreamCreate(&d_stream);
}

work_return_code_t copy_cuda::work(std::vector<block_work_input_sptr>& work_input,
                                   std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0]->items<uint8_t>();
    auto out = work_output[0]->items<uint8_t>();

    auto noutput_items = work_output[0]->n_items;
    auto itemsize = work_output[0]->buffer->item_size();
    checkCudaErrors(cudaMemcpyAsync(
        out, in, noutput_items * itemsize, cudaMemcpyDeviceToDevice, d_stream));

    cudaStreamSynchronize(d_stream);

    // Tell runtime system how many output items we produced.
    produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}
} // namespace streamops
} // namespace gr
