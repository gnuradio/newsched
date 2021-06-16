/* -*- c++ -*- */
/*
 * Copyright 2014,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/dtv/atsc_fpll.hh>

namespace gr {
namespace dtv {

atsc_fpll::sptr atsc_fpll::make(float rate)
{
    return std::make_shared<atsc_fpll>(rate);
}

atsc_fpll::atsc_fpll(float rate)
    : sync_block("dtv_atsc_fpll")
{
    add_port(port<gr_complex>::make("in", port_direction_t::INPUT));
    add_port(port<float>::make("out", port_direction_t::OUTPUT));

    d_afc.set_taps(1.0 - exp(-1.0 / rate / 5e-6));
    d_nco.set_freq((-3e6 + 0.309e6) / rate * 2 * GR_M_PI);
    d_nco.set_phase(0.0);
}

work_return_code_t atsc_fpll::work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output)
{
    constexpr float alpha = 0.01;
    constexpr float beta = alpha * alpha / 4.0;

    auto in = static_cast<const gr_complex*>(work_input[0].items());
    auto out = static_cast<float*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;

    float a_cos, a_sin;
    float x;
    gr_complex result, filtered;

    for (int k = 0; k < noutput_items; k++) {
        d_nco.step();                 // increment phase
        d_nco.sincos(&a_sin, &a_cos); // compute cos and sin

        // Mix out carrier and output I-only signal
        gr::fast_cc_multiply(result, in[k], gr_complex(a_sin, a_cos));

        out[k] = result.real();

        // Update phase/freq error
        filtered = d_afc.filter(result);
        x = gr::fast_atan2f(filtered.imag(), filtered.real());

        // std::cout << out[k] << "/" << x << std::endl;

        // avoid slamming filter with big transitions
        if (x > M_PI_2)
            x = M_PI_2;
        else if (x < -M_PI_2)
            x = -M_PI_2;

        d_nco.adjust_phase(alpha * x);
        d_nco.adjust_freq(beta * x);
    }

    work_output[0].n_produced = noutput_items;
    return work_return_code_t::WORK_OK;
}

} /* namespace dtv */
} /* namespace gr */
