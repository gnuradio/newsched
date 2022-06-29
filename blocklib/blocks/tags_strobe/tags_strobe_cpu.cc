/* -*- c++ -*- */
/*
 * Copyright 2013 Free Software Foundation, Inc.
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "tags_strobe_cpu.h"
#include "tags_strobe_cpu_gen.h"

namespace gr {
namespace blocks {

tags_strobe_cpu::tags_strobe_cpu(block_args args)
    : INHERITED_CONSTRUCTORS,
      d_nsamps(args.nsamps),
      d_tag(0, pmtf::map{ { args.key, args.value } })
{
}

work_return_code_t tags_strobe_cpu::work(std::vector<block_work_input_sptr>& work_input,
                                        std::vector<block_work_output_sptr>& work_output)
{
    auto optr = work_output[0]->raw_items();
    auto itemsize = work_output[0]->buffer->item_size();
    auto noutput_items = work_output[0]->n_items;
    memset(optr, 0, noutput_items * itemsize);

    uint64_t nitems =
        static_cast<uint64_t>(noutput_items) + work_output[0]->nitems_written();
    while ((nitems - d_offset) > d_nsamps) {
        d_offset += d_nsamps;
        d_tag.set_offset(d_offset);
        work_output[0]->add_tag(d_tag);
    }

    produce_each(noutput_items, work_output);
    return work_return_code_t::WORK_OK;
}


} // namespace blocks
} // namespace gr