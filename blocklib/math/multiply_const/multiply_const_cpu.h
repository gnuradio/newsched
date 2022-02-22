/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/math/multiply_const.h>

namespace gr {
namespace math {

template <class T>
class multiply_const_cpu : public multiply_const<T>
{
public:
    multiply_const_cpu(const typename multiply_const<T>::block_args& args);

    virtual work_return_code_t
    work(std::vector<block_work_input_sptr>& work_input,
         std::vector<block_work_output_sptr>& work_output) override;

protected:
    T d_k;
    size_t d_vlen;
};


} // namespace math
} // namespace gr
