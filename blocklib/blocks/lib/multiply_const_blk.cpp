/* -*- c++ -*- */
/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "multiply_const_blk.hpp"
#include <gnuradio/scheduler.hpp>
#include <volk/volk.h>

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

    add_param(param<T>(multiply_const<T>::params::id_k, "k", 1.0));

    add_param(param<size_t>(multiply_const<T>::params::id_vlen, "vlen", 1));
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
void multiply_const<T>::on_parameter_change(param_action_base param)
{
    if (param.id() == multiply_const<T>::params::id_k) {
        d_k = static_cast<param_action<T>>(param).new_value();
    } else if (param.id() == multiply_const<T>::params::id_vlen) {
        // cannot be changed
    }
}

template <class T>
void multiply_const<T>::on_parameter_query(param_action_base& param)
{
    if (param.id() == multiply_const<T>::params::id_k) {
        param.set_any_value(std::make_any<T>(d_k));
    } else if (param.id() == multiply_const<T>::params::id_vlen) {
        param.set_any_value(std::make_any<T>(d_vlen));
    }
}

template <class T>
void multiply_const<T>::set_k(T k)
{
    // call back to the scheduler if ptr is not null
    if (p_scheduler) {
        p_scheduler->request_parameter_change(
            alias(), param_action<T>(params::id_k, k, 0), [&](auto a) {
                std::cout << "k was changed to "
                          << static_cast<param_action<T>>(a).new_value() << std::endl;
            });

    }
    // else go ahead and update parameter value
    else {
        on_parameter_change(param_action<T>(params::id_k, k, 0));
    }
}

template <class T>
T multiply_const<T>::k()
{

    // call back to the scheduler if ptr is not null
    if (p_scheduler) {

        p_scheduler->request_parameter_query(
            alias(), param_action<T>(params::id_k, 0, 0), [&](auto a) {
                std::cout << "k was queried "
                          << static_cast<param_action<T>>(a).new_value() << std::endl;
            });

    }
    // else go ahead and return parameter value
    else {
        return d_k;
    }

    return d_k;
}

template class multiply_const<std::int16_t>;
template class multiply_const<std::int32_t>;
template class multiply_const<float>;
template class multiply_const<gr_complex>;
} /* namespace blocks */
} /* namespace gr */
