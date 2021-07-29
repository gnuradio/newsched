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

#include "interleaver_fifo.hh"

namespace gr {
namespace dtv {

/*!
 * \brief ATSC deinterleave RS encoded ATSC data ( atsc_mpeg_packet_rs_encoded -->
 * atsc_mpeg_packet_rs_encoded) \ingroup dtv_atsc
 *
 * input: atsc_mpeg_packet_rs_encoded; output: atsc_mpeg_packet_rs_encoded
 */
class atsc_deinterleaver : virtual public gr::sync_block
{
public:
    // gr::dtv::atsc_deinterleaver::sptr
    typedef std::shared_ptr<atsc_deinterleaver> sptr;

    /*!
     * \brief Make a new instance of gr::dtv::atsc_deinterleaver.
     */
    static sptr make();

    static constexpr int s_interleavers = 52;

    //! transform a single symbol
    unsigned char transform(unsigned char input)
    {
        unsigned char retval = m_fifo[m_commutator].stuff(input);
        m_commutator++;
        if (m_commutator >= s_interleavers)
            m_commutator = 0;
        return retval;
    }

    /*!
     * Note: The use of the alignment_fifo keeps the encoder and decoder
     * aligned if both are synced to a field boundary.  There may be other
     * ways to implement this function.  This is a best guess as to how
     * this should behave, as we have no test vectors for either the
     * interleaver or deinterleaver.
     */
    interleaver_fifo<unsigned char> alignment_fifo;

    int m_commutator;
    std::vector<interleaver_fifo<unsigned char>> m_fifo;

public:
    atsc_deinterleaver();
    ~atsc_deinterleaver() override;

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output) override;

    //! reset interleaver (flushes contents and resets commutator)
    void reset();

    //! sync interleaver (resets commutator, but doesn't flush fifos)
    void sync() { m_commutator = 0; }
};

} /* namespace dtv */
} /* namespace gr */
