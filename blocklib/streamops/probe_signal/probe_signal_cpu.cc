/* -*- c++ -*- */
/*
 * Copyright 2022 Josh Morman
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "probe_signal_cpu.h"
#include "probe_signal_cpu_gen.h"

namespace gr {
namespace streamops {

template <class T>
probe_signal_cpu<T>::probe_signal_cpu(const typename probe_signal<T>::block_args& args)
    : INHERITED_CONSTRUCTORS(T)
{
}

template <class T>
work_return_code_t probe_signal_cpu<T>::work(std::vector<block_work_input_sptr>& work_input,
                                         std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0]->items<T>();
    auto ninput_items = work_input[0]->n_items;

    if (ninput_items > 0)
        *this->param_level = in[ninput_items-1];

    this->consume_each(ninput_items, work_input);
    return work_return_code_t::WORK_OK;
}

} /* namespace streamops */
} /* namespace gr */
