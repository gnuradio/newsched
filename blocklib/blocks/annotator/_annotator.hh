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
 * \brief 1-to-1 stream annotator testing block. FOR TESTING PURPOSES ONLY.
 * \ingroup debug_tools_blk
 *
 * \details
 * This block creates tags to be sent downstream every 10,000
 * items it sees. The tags contain the name and ID of the
 * instantiated block, use "seq" as a key, and have a counter that
 * increments by 1 for every tag produced that is used as the
 * tag's value. The tags are propagated using the 1-to-1 policy.
 *
 * It also stores a copy of all tags it sees flow past it. These
 * tags can be recalled externally with the data() member.
 *
 * Warning: This block is only meant for testing and showing how to use the
 * tags.
 */
class annotator : virtual public sync_block
{
public:
    typedef std::shared_ptr<annotator> sptr;
    virtual std::vector<tag_t> data() const = 0;

    /**
     * @brief Set the implementation to CPU and return a shared pointer to the block instance
     * 
     * @return std::shared_ptr<annotator> 
     */
    static sptr make_cpu(uint64_t when, size_t itemsize, size_t num_inputs, size_t num_outputs, tag_propagation_policy_t tpp);
};

} /* namespace blocks */
} /* namespace gr */
