/* -*- c++ -*- */
/*
 * Copyright 2004,2008,2010,2013,2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include "vector_source.hpp"
#include <algorithm>
#include <stdexcept>

using namespace std;

namespace gr {
namespace blocks {

template <class T>
vector_source<T>::vector_source(const std::vector<T>& data,
                                bool repeat,
                                unsigned int vlen,
                                const std::vector<tag_t>& tags)
    : sync_block("vector_source"),
      d_data(data),
      d_repeat(repeat),
      d_offset(0),
      d_vlen(vlen),
      d_tags(tags)
{

    add_port(port<T>::make("output",
                               port_direction_t::OUTPUT,
                               port_type_t::STREAM,
                               std::vector<size_t>{ vlen }));

    add_param(param<std::vector<T>>(vector_source::params::data, "data", std::vector<T>()));
    add_param(param<bool>(vector_source::params::repeat, "repeat", false));

    if (tags.empty()) {
        d_settags = 0;
    } else {
        d_settags = 1;
        this->set_output_multiple(data.size() / vlen);
    }
    if ((data.size() % vlen) != 0)
        throw std::invalid_argument("data length must be a multiple of vlen");
}

template <class T>
void vector_source<T>::set_data(const std::vector<T>& data,
                                const std::vector<tag_t>& tags)
{
    d_data = data;
    d_tags = tags;
    rewind();
    if (tags.empty()) {
        d_settags = false;
    } else {
        d_settags = true;
    }
}

template <class T>
work_return_code_t vector_source<T>::work(std::vector<block_work_input>& work_input,
                                          std::vector<block_work_output>& work_output)
{

    // size_t noutput_ports = work_output.size(); // is 1 for this block
    int noutput_items = work_output[0].n_items;
    void* output_items = work_output[0].items;
    std::vector<tag_t> output_tags = work_output[0].tags;
    uint64_t n_written = work_output[0].n_items_written;

    T* optr = (T*)output_items;

    if (d_repeat) {
        unsigned int size = d_data.size();
        unsigned int offset = d_offset;
        if (size == 0)
            return work_return_code_t::WORK_DONE;

        if (d_settags) {
            int n_outputitems_per_vector = d_data.size() / d_vlen;
            for (int i = 0; i < noutput_items; i += n_outputitems_per_vector) {
                // FIXME do proper vector copy
                // memcpy((void*)optr, (const void*)&d_data[0], size * sizeof(T));
                std::copy( d_data.begin(), d_data.begin()+size, optr );
                optr += size;
                for (unsigned t = 0; t < d_tags.size(); t++) {

                    //   this->add_item_tag(0, this->nitems_written(0) + i +
                    //   d_tags[t].offset,
                    //                      d_tags[t].key, d_tags[t].value,
                    //                      d_tags[t].srcid);
                    output_tags.push_back(tag_t(n_written + i + d_tags[t].offset,
                                                d_tags[t].key,
                                                d_tags[t].value,
                                                d_tags[t].srcid));
                }
            }
        } else {
            for (int i = 0; i < static_cast<int>(noutput_items * d_vlen); i++) {
                optr[i] = d_data[offset++];
                if (offset >= size) {
                    offset = 0;
                }
            }
        }

        d_offset = offset;

        work_output[0].n_produced = noutput_items;
        return work_return_code_t::WORK_OK;

    } else {
        if (d_offset >= d_data.size())
            return work_return_code_t::WORK_DONE; // Done!

        unsigned n = std::min((unsigned)d_data.size() - d_offset,
                              (unsigned)noutput_items * d_vlen);
        for (unsigned i = 0; i < n; i++) {
            optr[i] = d_data[d_offset + i];
        }
        for (unsigned t = 0; t < d_tags.size(); t++) {
            if ((d_tags[t].offset >= d_offset) && (d_tags[t].offset < d_offset + n))
            {
                // this->add_item_tag(
                //     0, d_tags[t].offset, d_tags[t].key, d_tags[t].value, d_tags[t].srcid);
                work_output[0].tags.push_back(d_tags[t]);
            }
        }
        d_offset += n;

        work_output[0].n_produced = n / d_vlen;
        return work_return_code_t::WORK_OK;
    }
}

template class vector_source<std::uint8_t>;
template class vector_source<std::int16_t>;
template class vector_source<std::int32_t>;
template class vector_source<float>;
template class vector_source<gr_complex>;

} /* namespace blocks */
} /* namespace gr */
