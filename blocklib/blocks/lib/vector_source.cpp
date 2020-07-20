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
#include <cstring> // for memcpy

using namespace std;

namespace gr {
namespace blocks {

template <class T>
vector_source<T>::vector_source(const std::vector<T>& data,
                                bool repeat,
                                unsigned int vlen,
                                const std::vector<tag_t>& tags)
    : sync_block("vector_source"),
      _data(data),
      _repeat(repeat),
      _offset(0),
      _vlen(vlen),
      _tags(tags)
{

    if (tags.empty()) {
        _settags = 0;
    } else {
        _settags = 1;
        this->set_output_multiple(data.size() / vlen);
    }
    if ((data.size() % vlen) != 0)
        throw std::invalid_argument("data length must be a multiple of vlen");
}

template <class T>
void vector_source<T>::set_data(const std::vector<T>& data,
                                const std::vector<tag_t>& tags)
{
    _data = data;
    _tags = tags;
    rewind();
    if (tags.empty()) {
        _settags = false;
    } else {
        _settags = true;
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

    if (_repeat) {
        unsigned int size = _data.size();
        unsigned int offset = _offset;
        if (size == 0)
            return work_return_code_t::WORK_DONE;

        if (_settags) {
            int n_outputitems_per_vector = _data.size() / _vlen;
            for (int i = 0; i < noutput_items; i += n_outputitems_per_vector) {
                //std::copy(_data.begin(), _data.begin() + size, optr);
                memcpy((void*)optr, (const void*)&_data[0], size * sizeof(T));
                optr += size;
                for (unsigned t = 0; t < _tags.size(); t++) {
                    output_tags.push_back(tag_t(n_written + i + _tags[t].offset,
                                                _tags[t].key,
                                                _tags[t].value,
                                                _tags[t].srcid));
                }
            }
        } else {
            for (int i = 0; i < static_cast<int>(noutput_items * _vlen); i++) {
                optr[i] = _data[offset++];
                if (offset >= size) {
                    offset = 0;
                }
            }
        }

        _offset = offset;

        work_output[0].n_produced = noutput_items;
        return work_return_code_t::WORK_OK;

    } else {
        if (_offset >= _data.size())
            return work_return_code_t::WORK_DONE; // Done!

        unsigned n =
            std::min(_data.size() - _offset, noutput_items * _vlen);
        for (unsigned i = 0; i < n; i++) {
            optr[i] = _data[_offset + i];
        }
        for (unsigned t = 0; t < _tags.size(); t++) {
            if ((_tags[t].offset >= _offset) && (_tags[t].offset < _offset + n)) {
                // this->add_item_tag(
                //     0, _tags[t].offset, _tags[t].key, _tags[t].value,
                //     _tags[t].srcid);
                work_output[0].tags.push_back(_tags[t]);
            }
        }
        _offset += n;

        work_output[0].n_produced = n / _vlen;
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
