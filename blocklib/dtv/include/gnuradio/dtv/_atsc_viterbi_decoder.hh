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

#include <gnuradio/sync_block.hh>
#include <gnuradio/dtv/atsc_interleaver_fifo.hh>
#include "atsc_single_viterbi.hh"

namespace gr {
namespace dtv {

/*!
 * \brief ATSC Viterbi Decoder
 *
 * \ingroup dtv_atsc
 */
class atsc_viterbi_decoder : virtual public gr::sync_block
{
public:
    // gr::dtv::atsc_viterbi_decoder::sptr
    typedef std::shared_ptr<atsc_viterbi_decoder> sptr;

    /*!
     * \brief Make a new instance of gr::dtv::atsc_viterbi_decoder.
     */
    static sptr make();

    /*!
     * For each decoder, returns the current best state of the
     * decoding metrics.
     */
    std::vector<float> decoder_metrics() const;

    atsc_viterbi_decoder();

    void reset();

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output) override;

private:
    static const int NCODERS = 12;
    typedef interleaver_fifo<unsigned char> fifo_t;

    static constexpr int SEGMENT_SIZE = ATSC_MPEG_RS_ENCODED_LENGTH; // 207
    static constexpr int OUTPUT_SIZE = (SEGMENT_SIZE * 12);
    static constexpr int INPUT_SIZE = (ATSC_DATA_SEGMENT_LENGTH * 12);

    atsc_single_viterbi viterbi[NCODERS];
    std::vector<fifo_t> fifo;
};

} /* namespace dtv */
} /* namespace gr */
