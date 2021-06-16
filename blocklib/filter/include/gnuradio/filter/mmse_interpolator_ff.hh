/* -*- c++ -*- */
/*
 * Copyright 2004,2007,2012 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/block.hh>
#include <gnuradio/filter/mmse_fir_interpolator_ff.hh>

namespace gr {
namespace filter {

/*!
 * \brief Interpolating MMSE filter with float input, float output
 * \ingroup resamplers_blk
 */
class mmse_interpolator_ff : virtual public block
{
public:
    // gr::filter::mmse_interpolator_ff::sptr
    typedef std::shared_ptr<mmse_interpolator_ff> sptr;

    /*!
     * \brief Build the interpolating MMSE filter (float input, float output)
     *
     * \param phase_shift The phase shift of the output signal to the input
     * \param interp_ratio The interpolation ratio = input_rate / output_rate.
     */
    static sptr make(float phase_shift, float interp_ratio);

    float mu() const;
    float interp_ratio() const;
    void set_mu(float mu);
    void set_interp_ratio(float interp_ratio);


    mmse_interpolator_ff(float phase_shift, float interp_ratio);

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output) override;

private:
    double d_mu;
    double d_mu_inc;
    mmse_fir_interpolator_ff d_interp;

};

} /* namespace filter */
} /* namespace gr */
