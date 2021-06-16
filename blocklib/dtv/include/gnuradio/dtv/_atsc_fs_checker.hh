/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/block.hh>
#include "atsc_syminfo.hh"

namespace gr {
namespace dtv {

/*!
 * \brief ATSC Receiver FS_CHECKER
 *
 * \ingroup dtv_atsc
 */
class atsc_fs_checker : virtual public gr::block
{
public:
    // gr::dtv::atsc_fs_checker::sptr
    typedef std::shared_ptr<atsc_fs_checker> sptr;

    /*!
     * \brief Make a new instance of gr::dtv::atsc_fs_checker.
     */
    static sptr make();

    void reset();

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output) override;

    atsc_fs_checker();
    
private:
    

    static constexpr int SRSIZE = 1024; // must be power of two
    int d_index;                        // points at oldest sample
    float d_sample_sr[SRSIZE];          // sample shift register
    atsc::syminfo d_tag_sr[SRSIZE];     // tag shift register
    unsigned char d_bit_sr[SRSIZE];     // binary decision shift register
    int d_field_num;
    int d_segment_num;

    static constexpr int OFFSET_511 = 4;      // offset to second PN 63 pattern
    static constexpr int LENGTH_511 = 511;    // length of PN 63 pattern
    static constexpr int OFFSET_2ND_63 = 578; // offset to second PN 63 pattern
    static constexpr int LENGTH_2ND_63 = 63;  // length of PN 63 pattern

    inline static int wrap(int index) { return index & (SRSIZE - 1); }
    inline static int incr(int index) { return wrap(index + 1); }
    inline static int decr(int index) { return wrap(index - 1); }
};

} /* namespace dtv */
} /* namespace gr */
