/* -*- c++ -*- */
/*
 * Copyright 2012 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/streamops/interleaved_short_to_complex.hpp>
#include <volk/volk.h>

namespace gr {
namespace streamops {

interleaved_short_to_complex::sptr interleaved_short_to_complex::make(bool swap,
                                                                      float scale_factor)
{
    return std::make_shared<interleaved_short_to_complex>(swap, scale_factor);
}

interleaved_short_to_complex::interleaved_short_to_complex(bool swap, float scale_factor)
    : sync_block("interleaved_short_to_complex"), d_scalar(scale_factor), d_swap(swap)
{
    add_port(port<short>::make("in", port_direction_t::INPUT, { 2 }));
    add_port(port<gr_complex>::make("out", port_direction_t::OUTPUT, { 1 }));

    // const int alignment_multiple = volk_get_alignment() / sizeof(gr_complex);
    // set_alignment(std::max(1, alignment_multiple)); // TODO: output multiple
}

void interleaved_short_to_complex::set_swap(bool swap) { d_swap = swap; }

work_return_code_t
interleaved_short_to_complex::work(std::vector<block_work_input>& work_input,
                                   std::vector<block_work_output>& work_output)
{
    auto in = static_cast<const short*>(work_input[0].items());
    auto out = static_cast<float*>(work_output[0].items());

    auto noutput_items = work_output[0].n_items;

    // This calculates in[] * 1.0 / d_scalar
    volk_16i_s32f_convert_32f(out, in, d_scalar, 2 * noutput_items);

    if (d_swap) {
        for (int i = 0; i < noutput_items; ++i) {
            float f = out[2 * i + 1];
            out[2 * i + 1] = out[2 * i];
            out[2 * i] = f;
        }
    }

    work_output[0].n_produced = noutput_items;
    return work_return_code_t::WORK_OK;
}

} // namespace streamops
} // namespace gr
