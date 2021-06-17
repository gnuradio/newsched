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
#include "atsc_randomize.hh"

namespace gr {
namespace dtv {

/*!
 * \brief ATSC "dewhiten" incoming mpeg transport stream packets
 * \ingroup dtv_atsc
 *
 * input: atsc_mpeg_packet_no_sync; output: atsc_mpeg_packet;
 */
class atsc_derandomizer : virtual public gr::sync_block
{
public:
    // gr::dtv::atsc_derandomizer::sptr
    typedef std::shared_ptr<atsc_derandomizer> sptr;

    /*!
     * \brief Make a new instance of gr::dtv::atsc_derandomizer.
     */
    static sptr make();

private:
    atsc_randomize d_rand;

public:
    atsc_derandomizer();

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output) override;
};

} /* namespace dtv */
} /* namespace gr */
