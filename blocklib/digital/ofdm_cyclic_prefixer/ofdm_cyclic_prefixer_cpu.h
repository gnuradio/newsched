/* -*- c++ -*- */
/*
 * Copyright 2013, 2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/digital/ofdm_cyclic_prefixer.h>

namespace gr {
namespace digital {

class ofdm_cyclic_prefixer_cpu : public virtual ofdm_cyclic_prefixer
{
public:
    ofdm_cyclic_prefixer_cpu(block_args args);
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    //! FFT length
    size_t d_fft_len;
    //!  State, that determines the current output length used.
    unsigned d_state = 0;
    //! Variable being initialized with the largest CP.
    size_t d_cp_max = 0;
    //! Variable being initialized with the smallest CP.
    size_t d_cp_min = std::numeric_limits<size_t>::max();
    //! Length of pulse rolloff in samples
    size_t d_rolloff_len;
    //! Vector, that holds different CP lengths
    std::vector<size_t> d_cp_lengths;
    //! Buffers the up-flank (at the beginning of the cyclic prefix)
    std::vector<float> d_up_flank;
    //! Buffers the down-flank (which trails the symbol)
    std::vector<float> d_down_flank;
    //! Vector, that holds tail of the predecessor symbol.
    std::vector<gr_complex> d_delay_line; // We do this explicitly to avoid outputting
                                          // zeroes (i.e. no history!)
};

} // namespace digital
} // namespace gr