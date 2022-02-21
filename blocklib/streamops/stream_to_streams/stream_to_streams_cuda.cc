/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "stream_to_streams_cuda.hh"
#include "stream_to_streams_cuda_gen.hh"
#include <gnuradio/helper_cuda.h>
#include <volk/volk.h>

namespace gr {
namespace streamops {

stream_to_streams_cuda::stream_to_streams_cuda(const block_args& args)
    : INHERITED_CONSTRUCTORS, d_nstreams(args.nstreams)
{
    d_out_items.resize(args.nstreams);
    cudaStreamCreate(&d_stream);
}

work_return_code_t
stream_to_streams_cuda::work(std::vector<block_work_input_sptr>& work_input,
                             std::vector<block_work_output_sptr>& work_output)
{
    auto noutput_items = work_output[0]->n_items;
    auto ninput_items = work_input[0]->n_items;
    size_t nstreams = work_output.size();
    auto itemsize = work_output[0]->buffer->item_size();

    if (!p_kernel) {
        p_kernel =
            std::make_shared<cusp::deinterleave>((int)d_nstreams, 1, (int)itemsize);
        p_kernel->set_stream(d_stream);
    }

    

    auto total_items = std::min(ninput_items / nstreams, (size_t)noutput_items);

    d_out_items = block_work_output::all_items(work_output);

    p_kernel->launch_default_occupancy(
        { work_input[0]->items<uint8_t>() }, d_out_items, itemsize * total_items * nstreams);
    cudaStreamSynchronize(d_stream);

    produce_each(total_items, work_output);
    consume_each(total_items * nstreams, work_input);
    return work_return_code_t::WORK_OK;
}


} /* namespace streamops */
} /* namespace gr */
