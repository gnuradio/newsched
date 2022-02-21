/* -*- c++ -*- */
/*
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/math/add.hh>

namespace gr {
namespace math {

template <class T>
class add_cpu : public add<T>
{
public:
    add_cpu(const typename add<T>::block_args& args);
    
    virtual work_return_code_t work(std::vector<block_work_input_sptr>& work_input,
                                    std::vector<block_work_output_sptr>& work_output) override;

private:
    const size_t d_vlen;
    const size_t d_nports;
};


} // namespace math
} // namespace gr
