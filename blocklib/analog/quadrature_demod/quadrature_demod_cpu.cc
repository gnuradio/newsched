/* -*- c++ -*- */
/*
 * Copyright 2004,2005,2010,2012 Free Software Foundation, Inc.
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "quadrature_demod_cpu.h"
#include "quadrature_demod_cpu_gen.h"

#include <gnuradio/kernel/math/math.h>
#include <volk/volk.h>

namespace gr {
namespace analog {

quadrature_demod_cpu::quadrature_demod_cpu(block_args args) : INHERITED_CONSTRUCTORS
{
    // FIXME: do volk alignment
}

work_return_code_t
quadrature_demod_cpu::work(std::vector<block_work_input_sptr>& work_input,
                           std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0]->items<gr_complex>();
    auto out = work_output[0]->items<float>();
    auto noutput_items = work_output[0]->n_items;
    auto ninput_items = work_input[0]->n_items;

    // because of the history requirement, input needs to be 1 more than what we produce
    auto to_produce = std::min(ninput_items - (d_history - 1), noutput_items);
    auto gain = pmtf::get_as<float>(*this->param_gain);

    std::vector<gr_complex> tmp(to_produce);
    volk_32fc_x2_multiply_conjugate_32fc(&tmp[0], &in[1], &in[0], to_produce);
    for (size_t i = 0; i < to_produce; i++) {
        out[i] = gain * gr::kernel::math::fast_atan2f(imag(tmp[i]), real(tmp[i]));
    }

    produce_each(to_produce, work_output);
    consume_each(to_produce, work_input);
    return work_return_code_t::WORK_OK;
}


} // namespace analog
} // namespace gr