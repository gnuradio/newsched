/* -*- c++ -*- */
/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */


#ifndef INCLUDED_IO_SIGNATURE_H
#define INCLUDED_IO_SIGNATURE_H

#include <iostream>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace gr {
// If we use the capability publishing, it is too complicated to keep track
// of sizes and types - this may go away altogether
class io_signature_capability
{
private:
    int d_min_streams;
    int d_max_streams;

public:
    io_signature_capability(const int min_streams, const int max_streams);
    ~io_signature_capability() {}
};


class io_signature
{
private:
    int d_n_streams;
    std::vector<size_t> d_sizeof_stream_item;
    std::vector<std::type_index> d_type_info;

public:
    static constexpr int IO_INFINITE = -1;

    int n_streams() { return d_n_streams; }
    int max_streams() { return d_n_streams; }
    // io_signature(int n_streams,
    //              const std::vector<int>& sizeof_stream_items,  // size == n_streams
    //              std::vector<std::type_index>& type_index);    // size == n_streams -->
    //              get rid of n_streams

    // TODO: make the io_signature just take in the type information and vector lens??
    // io_signature(std::vector<std::type_index>& type_index)
    // {
    typedef std::shared_ptr<io_signature> sptr;
    io_signature() = default;
    io_signature(const std::vector<size_t>& sizeof_stream_items)
    {
        d_n_streams = sizeof_stream_items.size();
        d_sizeof_stream_item = sizeof_stream_items;
    }; // size == n_streams


    // }

    virtual ~io_signature() {}


    int sizeof_stream_item(int _index) const
    {
        if (_index < 0)
            throw std::invalid_argument("gr::io_signature::sizeof_stream_item");

        size_t index = _index;
        return d_sizeof_stream_item[std::min(index, d_sizeof_stream_item.size() - 1)];
    }

    std::vector<size_t> sizeof_stream_items() const { return d_sizeof_stream_item; }
};

typedef io_signature::sptr io_signature_sptr;


} // namespace gr

#endif