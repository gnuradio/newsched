/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "add_cuda.hh"
#include "add_cuda_gen.hh"
#include <volk/volk.h>

namespace gr {
namespace math {

template <class T>
add_cuda<T>::add_cuda(const typename add<T>::block_args& args)
    : sync_block("add_cuda"), add<T>(args), d_vlen(args.vlen), d_nports(args.nports)
{
    d_in_items.resize(d_nports);
    p_add_kernel =
        std::make_shared<cusp::add<T>>(d_nports);
}

template <class T>
work_return_code_t
add_cuda<T>::work(std::vector<block_work_input_sptr>& work_input,
                            std::vector<block_work_output_sptr>& work_output)
{
    auto out = work_output[0]->items<T>();
    auto noutput_items = work_output[0]->n_items;
    int noi = d_vlen * noutput_items;

    size_t idx = 0;
    for (auto& wi : work_input)
    {
        d_in_items[idx++] = wi->items<T>();
    }

    p_add_kernel->launch_default_occupancy(d_in_items, { out }, noi);

    this->produce_each(noutput_items, work_output);
    this->consume_each(noutput_items, work_input);
    return work_return_code_t::WORK_OK;

}

} /* namespace math */
} /* namespace gr */
