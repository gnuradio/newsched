/* -*- c++ -*- */
/*
 * Copyright <COPYRIGHT_YEAR> <COPYRIGHT_AUTHOR>
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "newblock_cpu.hh"
#include "newblock_cpu_gen.hh"

namespace gr {
namespace newmod {

template <class T>
newblock_cpu<T>::newblock_cpu(const typename newblock<T>::block_args& args)
    : INHERITED_CONSTRUCTORS(T)
{
}

template <class T>
work_return_code_t
newblock_cpu<T>::work(std::vector<block_work_input_sptr>& work_input,
                            std::vector<block_work_output_sptr>& work_output)
{
    // Do work specific code here
    return work_return_code_t::WORK_OK;
}

} /* namespace newmod */
} /* namespace gr */
