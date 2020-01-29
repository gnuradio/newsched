/* -*- c++ -*- */
/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */


#ifndef INCLUDED_IO_SIGNATURE_H
#define INCLUDED_IO_SIGNATURE_H

#include <iostream>
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
    std::vector<int> d_sizeof_stream_item;
    std::vector<std::type_index> d_type_info;

public:
    io_signature(int n_streams,
                 const std::vector<int>& sizeof_stream_items,
                 std::vector<std::type_index&>& type_index);

    ~io_signature() {}
};
} // namespace gr

#endif