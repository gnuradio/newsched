/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "newblock_cpu.hh"
#include <volk/volk.h>

namespace gr {
namespace newmod {

template <class T>
typename newblock<T>::sptr newblock<T>::make_cpu(const block_args& args)
{
    return std::make_shared<newblock_cpu<T>>(args);
}

template <class T>
newblock_cpu<T>::newblock_cpu(const typename newblock<T>::block_args& args)
    : sync_block("newblock"), newblock<T>(args)
{
}

template <class T>
work_return_code_t
newblock_cpu<T>::work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output)
{
    // Do work specific code here
    return work_return_code_t::WORK_OK;
}

template class newblock<std::int16_t>;
template class newblock<std::int32_t>;
template class newblock<float>;
template class newblock<gr_complex>;

} /* namespace newmod */
} /* namespace gr */
