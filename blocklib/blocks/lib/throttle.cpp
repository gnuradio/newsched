/* -*- c++ -*- */
/*
 * Copyright 2005-2011 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/blocklib/blocks/throttle.hpp>
#include <cstring>
#include <limits>
#include <thread>


namespace gr {
namespace blocks {


throttle::throttle(size_t itemsize, double samples_per_second, bool ignore_tags)
    : sync_block("throttle"),
      d_itemsize(itemsize),
      d_ignore_tags(ignore_tags)
{
    // TODO: make the throttle "don't care" type (size only)
    add_port(port<float>::make("input",
                               port_direction_t::INPUT,
                               port_type_t::STREAM,
                               std::vector<size_t>{ 1 }));
    add_port(port<float>::make("output",
                               port_direction_t::OUTPUT,
                               port_type_t::STREAM,
                               std::vector<size_t>{ 1 }));

    set_sample_rate(samples_per_second);
}

throttle::~throttle() {}

bool throttle::start()
{
    d_start = std::chrono::steady_clock::now();
    d_total_samples = 0;
    return block::start();
}

void throttle::set_sample_rate(double rate)
{
    // changing the sample rate performs a reset of state params
    d_start = std::chrono::steady_clock::now();
    d_total_samples = 0;
    d_sample_rate = rate;
    d_sample_period = std::chrono::duration<double>(1 / rate);
}

double throttle::sample_rate() const { return d_sample_rate; }

work_return_code_t throttle::work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output)
{
    // check for updated rx_rate tag
    if (!d_ignore_tags) {
        uint64_t abs_N = work_input[0].n_items_read;
        // std::vector<tag_t> all_tags;
        // get_tags_in_range(all_tags, 0, abs_N, abs_N + noutput_items);
        // for (const auto& tag : all_tags) {
        //     if (pmt::eq(tag.key, throttle_rx_rate_pmt)) {
        //         double new_rate = pmt::to_double(tag.value);
        //         set_sample_rate(new_rate);
        //     }
        // }
    }

    // copy all samples output[i] <= input[i]
    const char* in = (const char*)work_input[0].items;
    char* out = (char*)work_output[0].items;
    auto noutput_items = work_output[0].n_items;

    std::memcpy(out, in, noutput_items * d_itemsize);
    d_total_samples += noutput_items;

    auto now = std::chrono::steady_clock::now();
    auto expected_time = d_start + d_sample_period * d_total_samples;

    if (expected_time > now) {
        auto limit_duration =
            std::chrono::duration<double>(std::numeric_limits<long>::max());
        // if (expected_time - now > limit_duration) {
        //     GR_LOG_ALERT(d_logger,
        //                  "WARNING: Throttle sleep time overflow! You "
        //                  "are probably using a very low sample rate.");
        // }
        std::this_thread::sleep_until(expected_time);
    }

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}



} /* namespace blocks */
} /* namespace gr */
