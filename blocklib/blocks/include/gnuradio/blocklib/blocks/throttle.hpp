/* -*- c++ -*- */
/*
 * Copyright 2005-2011,2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_GR_THROTTLE_H
#define INCLUDED_GR_THROTTLE_H

#include <gnuradio/sync_block.hpp>
#include <chrono>

namespace gr {
namespace blocks {

/*!
 * \brief throttle flow of samples such that the average rate does
 * not exceed samples_per_sec.
 * \ingroup misc_blk
 *
 * \details
 * input: one stream of itemsize; output: one stream of itemsize
 *
 * N.B. this should only be used in GUI apps where there is no
 * other rate limiting block. It is not intended nor effective at
 * precisely controlling the rate of samples. That should be
 * controlled by a source or sink tied to sample clock. E.g., a
 * USRP or audio card.
 */
class throttle : virtual public sync_block
{
public:
    typedef std::shared_ptr<throttle> sptr;

    static sptr make(size_t itemsize, double samples_per_sec, bool ignore_tags = true)
    {
        auto ptr =
            std::make_shared<throttle>(itemsize, samples_per_sec, ignore_tags);

        ptr->add_port(untyped_port::make(
            "input", port_direction_t::INPUT, itemsize, port_type_t::STREAM));


        ptr->add_port(untyped_port::make(
            "output", port_direction_t::OUTPUT, itemsize, port_type_t::STREAM));

        return ptr;
    }

    throttle(size_t itemsize, double samples_per_sec, bool ignore_tags = true);
    // ~throttle();

    bool start();
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output);

    //! Sets the sample rate in samples per second.
    void set_sample_rate(double rate);

    //! Get the sample rate in samples per second.
    double sample_rate() const;

private:
    std::chrono::time_point<std::chrono::steady_clock> d_start;
    const size_t d_itemsize;
    uint64_t d_total_samples;
    double d_sample_rate;
    std::chrono::duration<double> d_sample_period;
    const bool d_ignore_tags;
};

} /* namespace blocks */
} /* namespace gr */

#endif /* INCLUDED_GR_THROTTLE_H */
