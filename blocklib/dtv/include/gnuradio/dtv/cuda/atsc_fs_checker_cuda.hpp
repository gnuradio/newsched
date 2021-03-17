/* -*- c++ -*- */
/*
 * Copyright 2014 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/block.hpp>
#include <gnuradio/dtv/atsc_syminfo.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "helper_cuda.h"

namespace gr {
namespace dtv {

/*!
 * \brief ATSC Receiver FS_CHECKER
 *
 * \ingroup dtv_atsc
 */
class atsc_fs_checker_cuda : virtual public gr::block
{
public:
    // gr::dtv::atsc_fs_checker::sptr
    typedef std::shared_ptr<atsc_fs_checker_cuda> sptr;

    /*!
     * \brief Make a new instance of gr::dtv::atsc_fs_checker.
     */
    static sptr make();

    void reset();

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output) override;

    atsc_fs_checker_cuda();
    
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
    static const int OUTPUT_MULTIPLE = 1;

    size_t d_max_output_items =  8; // 32768 / 4 / 828 // TODO - handle larger max_output items - limitation is in calculating downstream block sizes
    float* d_host_in;
    float* d_dev_in;
    uint8_t* d_dev_tmp;
    uint16_t* d_dev_nerrors511;
    uint16_t* d_dev_nerrors63;
    uint16_t* d_host_nerrors511;
    uint16_t* d_host_nerrors63;
    uint8_t* d_dev_atsc_pn511;
    uint8_t* d_dev_atsc_pn63;

    cudaStream_t stream1, stream2;

};

} /* namespace dtv */
} /* namespace gr */
