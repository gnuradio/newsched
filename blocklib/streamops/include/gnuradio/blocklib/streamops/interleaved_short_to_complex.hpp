/* -*- c++ -*- */
/*
 * Copyright 2012 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/sync_block.hpp>

namespace gr {
namespace streamops {

/*!
 * \brief Convert stream of interleaved shorts to a stream of complex
 * \ingroup type_converters_blk
 */
class interleaved_short_to_complex : virtual public sync_block
{
public:
    // gr::blocks::interleaved_short_to_complex::sptr
    typedef std::shared_ptr<interleaved_short_to_complex> sptr;

    /*!
     * Build an interleaved short to complex block.
     */
    static sptr
    make(bool swap = false, float scale_factor = 1.0f);

    void set_swap(bool swap);

    void set_scale_factor(float new_value) { d_scalar = new_value; };

    interleaved_short_to_complex(bool swap = false, float scale_factor = 1.0f);

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output) override;

private:
    float d_scalar;
    bool d_swap;
};

} // namespace streamops
} /* namespace gr */
