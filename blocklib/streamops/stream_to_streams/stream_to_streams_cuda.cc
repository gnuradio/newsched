/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "stream_to_streams_cuda.hh"
#include <gnuradio/helper_cuda.h>
#include <volk/volk.h>

namespace gr {
namespace streamops {


typename stream_to_streams::sptr stream_to_streams::make_cuda(const block_args& args)
{
    return std::make_shared<stream_to_streams_cuda>(args);
}

stream_to_streams_cuda::stream_to_streams_cuda(const block_args& args)
    : block("stream_to_streams_cuda"), stream_to_streams(args), d_itemsize(args.itemsize)
{
    d_out_items.resize(args.nstreams);
    p_kernel =
        std::make_shared<cusp::deinterleave>((int)args.nstreams, 1, (int)args.itemsize);

    cudaStreamCreate(&d_stream);
    p_kernel->set_stream(d_stream);
}

work_return_code_t
stream_to_streams_cuda::work(std::vector<block_work_input>& work_input,
                             std::vector<block_work_output>& work_output)
{
    auto noutput_items = work_output[0].n_items;
    auto ninput_items = work_input[0].n_items;
    size_t nstreams = work_output.size();

    auto total_items = std::min(ninput_items / nstreams, (size_t)noutput_items);

    d_out_items = block_work_output::all_items(work_output);

    p_kernel->launch_default_occupancy(
        { work_input[0].items<uint8_t>() }, d_out_items, d_itemsize * total_items * nstreams);
    cudaStreamSynchronize(d_stream);

    produce_each(total_items, work_output);
    consume_each(total_items * nstreams, work_input);
    return work_return_code_t::WORK_OK;
}


} /* namespace streamops */
} /* namespace gr */
