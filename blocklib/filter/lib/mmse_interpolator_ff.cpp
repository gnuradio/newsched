/* -*- c++ -*- */
/*
 * Copyright 2004,2007,2010 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/filter/mmse_interpolator_ff.hpp>
#include <stdexcept>

namespace gr {
namespace filter {

mmse_interpolator_ff::sptr mmse_interpolator_ff::make(float phase_shift,
                                                      float interp_ratio)
{
    return std::make_shared<mmse_interpolator_ff>(phase_shift, interp_ratio);
}

mmse_interpolator_ff::mmse_interpolator_ff(float phase_shift, float interp_ratio)
    : block("mmse_interpolator_ff"), d_mu(phase_shift), d_mu_inc(interp_ratio)
{

    add_port(port<float>::make("in", port_direction_t::INPUT));
    add_port(port<float>::make("out", port_direction_t::OUTPUT));

    GR_LOG_WARN(_logger,
                "mmse_interpolator is deprecated. Please use mmse_resampler instead.");

    if (interp_ratio <= 0)
        throw std::out_of_range("interpolation ratio must be > 0");
    if (phase_shift < 0 || phase_shift > 1)
        throw std::out_of_range("phase shift ratio must be > 0 and < 1");

    // set_inverse_relative_rate(d_mu_inc); // FIXME: not sure what to do with this just
    // yet
}

// void mmse_interpolator_ff::forecast(int noutput_items,
//                                     gr_vector_int& ninput_items_required)
// {
//     unsigned ninputs = ninput_items_required.size();
//     for (unsigned i = 0; i < ninputs; i++) {
//         ninput_items_required[i] =
//             (int)ceil((noutput_items * d_mu_inc) + d_interp.ntaps());
//     }
// }

work_return_code_t mmse_interpolator_ff::work(std::vector<block_work_input>& work_input,
                                              std::vector<block_work_output>& work_output)
{
    auto in = (const float*)work_input[0].items();
    auto out = (float*)work_output[0].items();
    auto noutput_items = work_output[0].n_items;
    int min_input_items = (int)ceil((noutput_items * d_mu_inc) + d_interp.ntaps());

    auto ninput_items = work_input[0].n_items;
    if (ninput_items < min_input_items) {
        return work_return_code_t::WORK_INSUFFICIENT_INPUT_ITEMS;
    }

    int ii = 0; // input index
    int oo = 0; // output index

    while (oo < noutput_items) {
        out[oo++] = d_interp.interpolate(&in[ii], static_cast<float>(d_mu));

        double s = d_mu + d_mu_inc;
        double f = floor(s);
        int incr = (int)f;
        d_mu = s - f;
        ii += incr;
    }

    work_input[0].n_consumed = ii;
    work_output[0].n_produced = noutput_items;

    return work_return_code_t::WORK_OK;
}

float mmse_interpolator_ff::mu() const { return static_cast<float>(d_mu); }

float mmse_interpolator_ff::interp_ratio() const { return static_cast<float>(d_mu_inc); }

void mmse_interpolator_ff::set_mu(float mu) { d_mu = static_cast<double>(mu); }

void mmse_interpolator_ff::set_interp_ratio(float interp_ratio)
{
    d_mu_inc = static_cast<double>(interp_ratio);
}

} /* namespace filter */
} /* namespace gr */
