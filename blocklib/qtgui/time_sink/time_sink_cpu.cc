/* -*- c++ -*- */
/*
 * Copyright 2004,2009,2010,2012,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "time_sink_cpu.hh"
#include <volk/volk.h>

namespace gr {
namespace qtgui {

template <class T>
typename time_sink<T>::sptr time_sink<T>::make_cpu(const block_args& args)
{
    return std::make_shared<time_sink_cpu<T>>(args);
}

template <class T>
time_sink_cpu<T>::time_sink_cpu(const typename time_sink<T>::block_args& args)
    : time_sink<T>(args)
{
}

template <>
work_return_code_t
time_sink_cpu<float>::work(std::vector<block_work_input>& work_input,
                                std::vector<block_work_output>& work_output)
{

    // Do block specific code here
    return work_return_code_t::WORK_OK;
}

template <>
work_return_code_t
time_sink_cpu<gr_complex>::work(std::vector<block_work_input>& work_input,
                                     std::vector<block_work_output>& work_output)
{
    // Do block specific code here
    return work_return_code_t::WORK_OK;
}

template <class T>
work_return_code_t
time_sink_cpu<T>::work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output)
{
    // Do work specific code here
    return work_return_code_t::WORK_OK;
}

template class time_sink<std::int16_t>;
template class time_sink<std::int32_t>;
template class time_sink<float>;
template class time_sink<gr_complex>;

} /* namespace qtgui */
} /* namespace gr */
