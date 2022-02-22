/* -*- c++ -*- */
/*
 * Copyright 2011,2012 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/filter/dc_blocker.h>

#include <cstdio>
#include <memory>

namespace gr {
namespace filter {

template <class T>
moving_averager<T>::moving_averager(int D)
    : d_length(D), d_out(0), d_out_d1(0), d_out_d2(0), d_delay_line(d_length - 1, 0)
{
}

template <class T>
T moving_averager<T>::filter(T x)
{
    d_out_d1 = d_out;
    d_delay_line.push_back(x);
    d_out = d_delay_line[0];
    d_delay_line.pop_front();

    T y = x - d_out_d1 + d_out_d2;
    d_out_d2 = y;

    return (y / (T)(d_length));
}

template <class T>
typename dc_blocker<T>::sptr dc_blocker<T>::make(int D, bool long_form)
{
    return std::make_shared<dc_blocker<T>>(D, long_form);
}

template <class T>
dc_blocker<T>::dc_blocker(int D, bool long_form)
    : d_length(D), d_long_form(long_form), d_ma_0(D), d_ma_1(D)
{
    add_port(port<T>::make("in", port_direction_t::INPUT, { 1 }));
    add_port(port<T>::make("out", port_direction_t::OUTPUT, { 1 }));

    if (d_long_form) {
        d_ma_2 = std::make_unique<moving_averager<T>>(D);
        d_ma_3 = std::make_unique<moving_averager<T>>(D);
        d_delay_line = std::deque<T>(d_length - 1, 0);
    }
}

template <class T>
int dc_blocker<T>::group_delay()
{
    if (d_long_form)
        return (2 * d_length - 2);
    else
        return d_length - 1;
}

template <class T>
work_return_code_t dc_blocker<T>::work(std::vector<block_work_input_sptr>& work_input,
                                       std::vector<block_work_output_sptr>& work_output)
{
    auto in = work_input[0]->items<T>());
    auto out = work_output[0]->items<T>());
    auto noutput_items = work_output[0]->n_items;

    if (d_long_form) {
        T y1, y2, y3, y4, d;
        for (int i = 0; i < noutput_items; i++) {
            y1 = d_ma_0.filter(in[i]);
            y2 = d_ma_1.filter(y1);
            y3 = d_ma_2->filter(y2);
            y4 = d_ma_3->filter(y3);

            d_delay_line.push_back(d_ma_0.delayed_sig());
            d = d_delay_line[0];
            d_delay_line.pop_front();

            out[i] = d - y4;
        }
    }
    else {
        T y1, y2;
        for (int i = 0; i < noutput_items; i++) {
            y1 = d_ma_0.filter(in[i]);
            y2 = d_ma_1.filter(y1);
            out[i] = d_ma_0.delayed_sig() - y2;
        }
    }

    work_output[0]->n_produced = noutput_items;
    return work_return_code_t::WORK_OK;
}

template class dc_blocker<float>;
template class dc_blocker<gr_complex>;

} /* namespace filter */
} /* namespace gr */
