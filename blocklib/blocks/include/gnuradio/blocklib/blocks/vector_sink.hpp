/* -*- c++ -*- */
/*
 * Copyright 2004,2008,2010,2013,2018,2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#ifndef INCLUDED_VECTOR_SINK_HPP
#define INCLUDED_VECTOR_SINK_HPP

#include <gnuradio/sync_block.hpp>

#include <cstdint>
#include <mutex>
namespace gr {
namespace blocks {


template <class T>
class vector_sink : virtual public sync_block
{

public:
    enum params : uint32_t { id_vlen, id_data, id_tags, num_params };
    typedef std::shared_ptr<vector_sink> sptr;

    static sptr make(const size_t vlen = 1, const size_t reserve_items = 1024)
    {
        auto ptr = std::make_shared<vector_sink>(vector_sink(vlen, reserve_items));

        ptr->add_port(port<T>::make("input",
                                    port_direction_t::INPUT,
                                    port_type_t::STREAM,
                                    std::vector<size_t>{ vlen }));

        ptr->add_param(param<size_t>::make(
            vector_sink::params::id_vlen, "vlen", vlen, &(ptr->_vlen)));
        ptr->add_param(param<std::vector<T>>::make(
            vector_sink::params::id_data, "data", std::vector<T>{}, &(ptr->_data)));
        ptr->add_param(param<std::vector<tag_t>>::make(
            vector_sink::params::id_tags, "tags", std::vector<tag_t>{}, &(ptr->_tags)));

        return ptr;
    }

    vector_sink(const size_t vlen = 1, const size_t reserve_items = 1024);
    // ~vector_sink() {};

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output);

    std::any handle_reset(std::vector<std::any> args)
    {
        _tags.clear();
        _data.clear();

        return std::any();
    }

    //! Clear the data and tags containers.
    void reset();
    std::vector<T> data()
    {
        return request_parameter_query<std::vector<T>>(params::id_data);
    }

    // since _data is changed inside work(), we must catch and update the current value
    virtual void on_parameter_query(param_action_sptr action)
    {
        if (action->id() == id_data) {
            auto param = parameters.get(action->id());
            action->set_any_value(std::any_cast<std::vector<T>>(_data));
        } else {
            block::on_parameter_query(action);
        }
    }

private:
    std::vector<T> _data;
    std::vector<tag_t> _tags;
    size_t _vlen;
};
typedef vector_sink<std::uint8_t> vector_sink_b;
typedef vector_sink<std::int16_t> vector_sink_s;
typedef vector_sink<std::int32_t> vector_sink_i;
typedef vector_sink<float> vector_sink_f;
typedef vector_sink<gr_complex> vector_sink_c;
} /* namespace blocks */
} /* namespace gr */

#endif