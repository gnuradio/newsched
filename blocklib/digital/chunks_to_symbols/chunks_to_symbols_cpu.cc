/* -*- c++ -*- */
/*
 * Copyright 2022 FIXME
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "chunks_to_symbols_cpu.h"
#include "chunks_to_symbols_cpu_gen.h"

namespace gr {
namespace digital {

template <class IN_T, class OUT_T>
chunks_to_symbols_cpu<IN_T, OUT_T>::chunks_to_symbols_cpu(
    const typename chunks_to_symbols<IN_T, OUT_T>::block_args& args)
    : INHERITED_CONSTRUCTORS(IN_T, OUT_T)
{
    this->set_output_multiple(args.D);
}

template <class IN_T, class OUT_T>
work_return_code_t
chunks_to_symbols_cpu<IN_T, OUT_T>::work(std::vector<block_work_input_sptr>& work_input,
                                         std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0]->items<IN_T>();
    auto out = work_output[0]->items<OUT_T>();

    auto noutput = work_output[0]->n_items;
    auto ninput = work_input[0]->n_items;

    auto d_D = pmtf::get_as<size_t>(*this->param_D);

    // number of inputs to consume
    auto in_count = std::min(ninput, noutput / d_D);
    if (in_count < 1) {
        return work_return_code_t::WORK_INSUFFICIENT_OUTPUT_ITEMS;
    }

    auto d_symbol_table = pmtf::get_as<std::vector<OUT_T>>(*this->param_symbol_table);

    if (d_D == 1) {
        for (size_t i = 0; i < in_count; i++) {
            auto key = static_cast<size_t>(*in);
            *out = d_symbol_table[key];
            ++out;
            ++in;
        }
    }
    else { // the multi-dimensional case
        for (size_t i = 0; i < in_count; i++) {
            auto key = static_cast<size_t>(*in) * d_D;
            for (size_t idx = 0; idx < d_D; ++idx) {
                *out = d_symbol_table[key + idx];
                ++out;
            }
            ++in;
        }
    }

    this->consume_each(in_count, work_input);
    this->produce_each(in_count * d_D, work_output);
    return work_return_code_t::WORK_OK;
}
} // namespace digital
} // namespace gr
