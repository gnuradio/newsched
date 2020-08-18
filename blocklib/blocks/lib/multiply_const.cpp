/* -*- c++ -*- */
/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "multiply_const.hpp"
#include <volk/volk.h>
#include <chrono>
#include <mutex>
#include <thread>

#include <gnuradio/scheduler.hpp>

namespace gr {
namespace blocks {

template <>
multiply_const<float>::multiply_const()
    : sync_block("multiply_const_ff")
{
    const int alignment_multiple = volk_get_alignment() / sizeof(float);
    set_alignment(std::max(1, alignment_multiple));
}

template <>
work_return_code_t
multiply_const<float>::work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output)
{

    const float* in = (const float*)work_input[0].items;
    float* out = (float*)work_output[0].items;
    int noi = work_output[0].n_items * d_vlen;

    volk_32f_s32f_multiply_32f(out, in, d_k, noi);

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}

template <>
multiply_const<gr_complex>::multiply_const()
    : sync_block("multiply_const_cc")
{
    const int alignment_multiple = volk_get_alignment() / sizeof(gr_complex);
    set_alignment(std::max(1, alignment_multiple));
}

template <>
work_return_code_t
multiply_const<gr_complex>::work(std::vector<block_work_input>& work_input,
                                 std::vector<block_work_output>& work_output)
{
    const gr_complex* in = (const gr_complex*)work_input[0].items;
    gr_complex* out = (gr_complex*)work_output[0].items;
    int noi = work_output[0].n_items * d_vlen;

    volk_32fc_s32fc_multiply_32fc(out, in, d_k, noi);

    work_output[0].n_produced = work_output[0].n_items;
    return work_return_code_t::WORK_OK;
}


template <class T>
multiply_const<T>::multiply_const()
    : sync_block("multiply_const")
{

}

template <class T>
work_return_code_t multiply_const<T>::work(std::vector<block_work_input>& work_input,
                                           std::vector<block_work_output>& work_output)
{
    // Pre-generate these from modtool, for example
    T* iptr = (T*)work_input[0].items;
    T* optr = (T*)work_output[0].items;

    int size = work_output[0].n_items * d_vlen;

    while (size >= 8) {
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        *optr++ = *iptr++ * d_k;
        size -= 8;
    }

    while (size-- > 0)
        *optr++ = *iptr++ * d_k;

    work_output[0].n_produced = work_output[0].n_items;
    work_input[0].n_consumed = work_input[0].n_items;
    return work_return_code_t::WORK_OK;
}

// A notional generic callback used as an example.  Should not actually be part of the
// multiply_const block
template <class T>
double multiply_const<T>::do_a_bunch_of_things(const int x,
                                               const double y,
                                               const std::vector<gr_complex>& z)
{
    // call back to the scheduler if ptr is not null
    if (p_scheduler) {
        bool cb_complete = false;
        int val;
        p_scheduler->request_callback(
            alias(),
            callback_args{
                "do_a_bunch_of_things",
                std::vector<std::any>{ std::make_any<int>(x),
                                       std::make_any<double>(y),
                                       std::make_any<std::vector<gr_complex>>(z) },
                std::any(),
                0 },
            [&cb_complete, &val](auto cb_args) {
                cb_complete = true;
                val = std::any_cast<double>(cb_args.return_val);
            });

        // block
        while (!cb_complete) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        gr_log_debug(_debug_logger, "callback returned {} ", val);

        return val;
    }
    // else go ahead and return parameter value
    else {
        return 0;
    }
}

template class multiply_const<std::int16_t>;
template class multiply_const<std::int32_t>;
template class multiply_const<float>;
template class multiply_const<gr_complex>;
} /* namespace blocks */
} /* namespace gr */
