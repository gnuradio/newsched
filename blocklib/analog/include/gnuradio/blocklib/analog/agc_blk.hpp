/* -*- c++ -*- */
/*
 * Copyright 2005,2006,2012 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/blocklib/analog/agc.hpp>
#include <gnuradio/sync_block.hpp>

namespace gr {
namespace analog {

/*!
 * \brief high performance Automatic Gain Control class
 * \ingroup level_controllers_blk
 *
 * \details
 * Power is approximated by absolute value
 */

template <class T>
class agc_blk : virtual public sync_block, kernel::agc<T>
{
public:
    // gr::analog::agc_ff::sptr
    typedef std::shared_ptr<agc_blk<T>> sptr;

    /*!
     * Build a floating point AGC loop block.
     *
     * \param rate the update rate of the loop.
     * \param reference reference value to adjust signal power to.
     * \param gain initial gain value.
     */
    static sptr make(float rate = 1e-4, float reference = 1.0, float gain = 1.0);

    void set_rate(float rate);
    void set_reference(float reference);
    void set_gain(float gain);
    void set_max_gain(float max_gain);

    agc_blk(float rate = 1e-4, float reference = 1.0, float gain = 1.0);

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output);
};

} /* namespace analog */
} /* namespace gr */
