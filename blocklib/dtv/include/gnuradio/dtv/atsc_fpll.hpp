/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/sync_block.hpp>
#include <gnuradio/math/nco.hpp>
#include <gnuradio/filter/single_pole_iir.hpp>

namespace gr {
namespace dtv {

/*!
 * \brief ATSC Receiver FPLL
 *
 * This block is takes in a complex I/Q baseband stream from the
 * receive filter and outputs the 8-level symbol stream.
 *
 * It does this by first locally generating a pilot tone and
 * complex mixing with the input signal.  This results in the
 * pilot tone shifting to DC and places the signal in the upper
 * sideband.
 *
 * As no information is encoded in the phase of the waveform, the
 * Q channel is then discarded, producing a real signal with the
 * lower sideband restored.
 *
 * The 8-level symbol stream still has a DC offset, and still
 * requires symbol timing recovery.
 *
 * \ingroup dtv_atsc
 */
class atsc_fpll : virtual public gr::sync_block
{
public:
    // gr::dtv::atsc_fpll::sptr
    typedef std::shared_ptr<atsc_fpll> sptr;

    /*!
     * \brief Make a new instance of gr::dtv::atsc_fpll.
     *
     * param rate  Sample rate of incoming stream
     */
    static sptr make(float rate);

    atsc_fpll(float rate);

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output) override;

private:
    gr::nco<float, float> d_nco;
    gr::filter::single_pole_iir<gr_complex, gr_complex, float> d_afc;

};

} /* namespace dtv */
} /* namespace gr */
