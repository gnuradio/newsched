/* -*- c++ -*- */
/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */


#ifndef INCLUDED_IO_SIGNATURE_H
#define INCLUDED_IO_SIGNATURE_H

#include <vector>
namespace gr {
class io_signature
{

private:
    int d_min_streams;
    int d_max_streams;
    std::vector<int> d_sizeof_stream_item;

    // TODO: type info


    io_signature(int min_streams,
                 int max_streams,
                 const std::vector<int>& sizeof_stream_items);



};
}

#endif