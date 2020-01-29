/* -*- c++ -*- */
/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */


#ifndef INCLUDED_IO_SIGNATURE_H
#define INCLUDED_IO_SIGNATURE_H

#include <vector>
#include <iostream>
#include <typeinfo>
#include <typeindex>

namespace gr {
class io_signature_capability
{

private:
    int d_min_streams;
    int d_max_streams;
    std::vector<int> d_sizeof_stream_item;
    std::vector<std::type_index&> d_type_index;

public:
    io_signature_capability(int min_streams,
                 int max_streams,
                 const std::vector<int>& sizeof_stream_items,
                 std::vector<std::type_index&>& type_index);

    ~io_signature_capability(){}

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

    ~io_signature(){}

};
}

#endif