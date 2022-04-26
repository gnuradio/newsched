/* -*- c++ -*- */
/*
 * Copyright 2022 FIXME
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/digital/chunks_to_symbols.h>

namespace gr {
namespace digital {

template <class IN_T, class OUT_T>
class chunks_to_symbols_cpu : public chunks_to_symbols<IN_T, OUT_T>
{
public:
    chunks_to_symbols_cpu(const typename chunks_to_symbols<IN_T, OUT_T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    // Declare private variables here
};


} // namespace digital
} // namespace gr
