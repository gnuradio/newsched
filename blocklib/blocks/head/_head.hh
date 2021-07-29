/* -*- c++ -*- */
/*
 * Copyright 2010,2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/sync_block.hh>

namespace gr {
namespace blocks {

/*!
 * \brief copies the first N items to the output then signals done
 * \ingroup misc_blk
 *
 * \details
 * Useful for building test cases
 */
class head : public sync_block
{
public:
    typedef std::shared_ptr<head> sptr;
    head(size_t itemsize) : sync_block("head")
    {
        add_port(untyped_port::make("input", port_direction_t::INPUT, itemsize));
        add_port(untyped_port::make("output", port_direction_t::OUTPUT, itemsize));
    }
    /**
     * @brief Set the implementation to CPU and return a shared pointer to the block
     * instance
     *
     * @return std::shared_ptr<head>
     */
    static sptr make_cpu(size_t itemsize, size_t nitems);
};

} /* namespace blocks */
} /* namespace gr */
