/* -*- c++ -*- */
/*
 * Copyright 2005,2006,2010,2012 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/blocklib/analog/agc_blk.hpp>

namespace gr {
namespace analog {

template <class T>
typename agc_blk<T>::sptr agc_blk<T>::make(float rate, float reference, float gain)
{
    return std::make_shared<agc_blk<T>>(rate, reference, gain);
}

template <class T>
agc_blk<T>::agc_blk(float rate, float reference, float gain)
    : sync_block("agc_blk"), kernel::agc<T>(rate, reference, gain, 65536)
{
    add_port(port<T>::make("in", port_direction_t::INPUT, { 1 }));
    add_port(port<T>::make("out", port_direction_t::OUTPUT, { 1 }));
}

template <class T>
void agc_blk<T>::set_rate(float rate)
{
    kernel::agc<T>::set_rate(rate);
}
template <class T>
void agc_blk<T>::set_reference(float reference)
{
    kernel::agc<T>::set_reference(reference);
}
template <class T>
void agc_blk<T>::set_gain(float gain)
{
    kernel::agc<T>::set_gain(gain);
}
template <class T>
void agc_blk<T>::set_max_gain(float max_gain)
{
    kernel::agc<T>::set_max_gain(max_gain);
}


template <class T>
work_return_code_t agc_blk<T>::work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output)
{

    auto in = static_cast<const T*>(work_input[0].items());
    auto out = static_cast<T*>(work_output[0].items());
    auto noutput_items = work_output[0].n_items;
    kernel::agc<T>::scaleN(out, in, noutput_items);

    work_output[0].n_produced = noutput_items;
    return work_return_code_t::WORK_OK;
}

template class agc_blk<float>;
template class agc_blk<gr_complex>;

} /* namespace analog */
} /* namespace gr */
