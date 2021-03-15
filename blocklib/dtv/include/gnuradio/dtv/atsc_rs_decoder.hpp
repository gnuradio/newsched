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

#include "atsc_types.hpp"
#include <gnuradio/sync_block.hpp>

extern "C" {
#include <gnuradio/fec/rs.h>
}


namespace gr {
namespace dtv {

/*!
 * \brief ATSC Receiver Reed-Solomon Decoder
 *
 * \ingroup dtv_atsc
 */
class atsc_rs_decoder : virtual public gr::sync_block
{
public:
    // gr::dtv::atsc_rs_decoder::sptr
    typedef std::shared_ptr<atsc_rs_decoder> sptr;

    /*!
     * Returns the number of errors corrected by the decoder.
     */
    int num_errors_corrected() const;

    /*!
     * Returns the number of bad packets rejected by the decoder.
     */
    int num_bad_packets() const;

    /*!
     * Returns the total number of packets seen by the decoder.
     */
    int num_packets() const;

    /*!
     * \brief Make a new instance of gr::dtv::atsc_rs_decoder.
     */
    static sptr make();

private:
    int d_nerrors_corrected_count;
    int d_bad_packet_count;
    int d_total_packets;
    void* d_rs;

public:
    atsc_rs_decoder();
    ~atsc_rs_decoder() override;

    /*!
     * Decode RS encoded packet.
     * \returns a count of corrected symbols, or -1 if the block was uncorrectible.
     */
    int decode(uint8_t* out, const uint8_t* in);

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output) override;
};

} /* namespace dtv */
} /* namespace gr */
