/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "pdu_to_stream_cpu.h"
#include "pdu_to_stream_cpu_gen.h"

#include <gnuradio/pdu.h>

namespace gr {
namespace pdu {

pdu_to_stream_cpu::pdu_to_stream_cpu(const typename pdu_to_stream::block_args& args)
    : INHERITED_CONSTRUCTORS
{
}

work_return_code_t
pdu_to_stream_cpu::work(std::vector<block_work_input_sptr>& work_input,
                           std::vector<block_work_output_sptr>& work_output)
{
    auto out = work_output[0]->items<uint8_t>();
    auto noutput_items = work_output[0]->n_items;
    int itemsize =  work_output[0]->buffer->item_size();

    // fill up the output buffer with the data from the pdus
    size_t i = 0;
    while (i < noutput_items) {
        if (!d_vec_ready && !d_pmt_queue.empty()) {
            d_pdu = pmtf::pdu(d_pmt_queue.front());
            d_pmt_queue.pop();
            d_vec_idx = 0;
            d_vec_ready = true;
        }

        if (d_vec_ready) {
            auto num_in_this_pmt = std::min(noutput_items - i, (d_pdu.size_bytes() - d_vec_idx) / itemsize );

            std::copy(d_pdu.raw() + d_vec_idx,
                      d_pdu.raw() + d_vec_idx + num_in_this_pmt * itemsize,
                      out + i * itemsize);
            i += num_in_this_pmt;
            d_vec_idx += num_in_this_pmt * itemsize;

            if (d_vec_idx >= d_pdu.size_bytes()) {
                d_vec_ready = false;
            }
        }
        else {
            break;
        }
    }

    this->produce_each(i, work_output);
    return work_return_code_t::WORK_OK;
}

void pdu_to_stream_cpu::handle_msg_pdus(pmtf::pmt msg)
{
    d_pmt_queue.push(msg);
    this->notify_scheduler_output();
}

} /* namespace pdu */
} /* namespace gr */
