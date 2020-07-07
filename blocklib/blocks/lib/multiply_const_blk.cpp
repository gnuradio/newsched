/* -*- c++ -*- */
/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "multiply_const_blk.hpp"
#include <gnuradio/scheduler.hpp>
#include <condition_variable>
#include <volk/volk.h>
#include <chrono>
#include <mutex>
#include <thread>

namespace gr {
namespace blocks {

template <class T>
void multiply_const<T>::ports_and_params(size_t vlen)
{
    add_port(port<T>::make("input",
                           port_direction_t::INPUT,
                           port_type_t::STREAM,
                           std::vector<size_t>{ vlen }));
    add_port(port<T>::make("output",
                           port_direction_t::OUTPUT,
                           port_type_t::STREAM,
                           std::vector<size_t>{ vlen }));

    add_param(param<T>::make(multiply_const<T>::params::id_k, "k", 1.0, &d_k));

    // TODO: vlen should be const and unchangeable as a parameter
    add_param(
        param<size_t>::make(multiply_const<T>::params::id_vlen, "vlen", vlen, &d_vlen));

    register_callback("do_a_bunch_of_things", [this](auto args) {
        return this->handle_do_a_bunch_of_things(args);
    });
}

template <>
multiply_const<float>::multiply_const(float k, size_t vlen)
    : sync_block("multiply_const_ff"), d_k(k), d_vlen(vlen)
{
    ports_and_params(vlen);
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
multiply_const<gr_complex>::multiply_const(gr_complex k, size_t vlen)
    : sync_block("multiply_const_cc"), d_k(k), d_vlen(vlen)
{
    ports_and_params(vlen);
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
multiply_const<T>::multiply_const(T k, size_t vlen)
    : sync_block("multiply_const"), d_k(k), d_vlen(vlen)
{
    ports_and_params(vlen);
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

template <class T>
void multiply_const<T>::set_k(T k)
{
    // call back to the scheduler if ptr is not null
    if (p_scheduler) {
        std::condition_variable cv;
        std::mutex m;
        auto lam = [&](param_action_sptr a) {
            std::unique_lock<std::mutex> lk(m);
            cv.notify_one();
        };
        p_scheduler->request_parameter_change(
            alias(), param_action<T>::make(params::id_k, k, 0), lam);

        // block until confirmation that parameter has been set
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk);

    }
    // else go ahead and update parameter value
    else {
        on_parameter_change(param_action<T>::make(params::id_k, k, 0));
    }
}

template <class T>
T multiply_const<T>::k()
{
    // call back to the scheduler if ptr is not null
    if (p_scheduler) {
        std::condition_variable cv;
        std::mutex m;
        T newval;
        auto lam = [&](param_action_sptr a) {
            std::unique_lock<std::mutex> lk(m);
            newval = std::static_pointer_cast<param_action<T>>(a)->new_value();
            cv.notify_one();
        };

        p_scheduler->request_parameter_query(
            alias(), param_action<T>::make(params::id_k, 0, 0), lam);

        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk);
        return newval;
    }
    // else go ahead and return parameter value
    else {
        return d_k;
    }
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

        std::cout << "callback returned " << val << std::endl;

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
