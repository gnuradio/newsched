/* -*- c++ -*- */
/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "multiply_const_blk.hpp"
#include <volk/volk.h>

namespace gr {
namespace blocks {

enum multiply_const_blk_params : uint32_t { k, vlen };

template <>
multiply_const<float>::multiply_const(float k, size_t vlen)
    : sync_block("multiply_const_ff"), d_k(k), d_vlen(vlen)
{
    add_port(typed_port<float>("input",
                               port_direction_t::INPUT,
                               port_type_t::STREAM,
                               std::vector<size_t>{ vlen }));
    add_port(typed_port<float>("output",
                               port_direction_t::OUTPUT,
                               port_type_t::STREAM,
                               std::vector<size_t>{ vlen }));

    add_param(typed_param<float>(multiply_const_blk_params::k,
                "k",
                1.0));

    add_param(typed_param<size_t>(multiply_const_blk_params::vlen,
                "vlen",
                1));

    std::cout << "mult constructor" << std::endl;
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
    add_port(typed_port<gr_complex>("input",
                               port_direction_t::INPUT,
                               port_type_t::STREAM,
                               std::vector<size_t>{ vlen }));
    add_port(typed_port<gr_complex>("output",
                               port_direction_t::OUTPUT,
                               port_type_t::STREAM,
                               std::vector<size_t>{ vlen }));

    add_param(typed_param<gr_complex>(multiply_const_blk_params::k,
                "k",
                1.0));

    add_param(typed_param<gr_complex>(multiply_const_blk_params::vlen,
                "vlen",
                1));

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
    add_port(typed_port<T>("input",
                               port_direction_t::INPUT,
                               port_type_t::STREAM,
                               std::vector<size_t>{ vlen }));
    add_port(typed_port<T>("output",
                               port_direction_t::OUTPUT,
                               port_type_t::STREAM,
                               std::vector<size_t>{ vlen }));

    add_param(typed_param<T>(multiply_const_blk_params::k,
                "k",
                1.0));

    add_param(typed_param<T>(multiply_const_blk_params::vlen,
                "vlen",
                1));
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
void multiply_const<T>::on_parameter_change(std::vector<block_param> params)
{
    for (auto &p : params)
    {
        if (p.id() == multiply_const_blk_params::k)
        {
            d_k = static_cast<typed_param<float>>(p).value();
        }
        else if (p.id() == multiply_const_blk_params::vlen)
        {
            // cannot be changed
        }
    }
}

template class multiply_const<std::int16_t>;
template class multiply_const<std::int32_t>;
template class multiply_const<float>;
template class multiply_const<gr_complex>;
} /* namespace blocks */
} /* namespace gr */
