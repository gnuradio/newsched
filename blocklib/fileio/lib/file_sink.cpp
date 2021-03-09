/* -*- c++ -*- */
/*
 * Copyright 2004,2006,2007,2010,2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <gnuradio/fileio/file_sink.hpp>
#include <stdexcept>

namespace gr {
namespace fileio {

file_sink::sptr file_sink::make(size_t itemsize, const char* filename, bool append)
{
    return std::make_shared<file_sink>(itemsize, filename, append);
}

file_sink::file_sink(size_t itemsize, const char* filename, bool append)
    : sync_block("file_sink"), file_sink_base(filename, true, append), d_itemsize(itemsize)
{

    add_port(untyped_port::make("input", port_direction_t::INPUT, itemsize));
}

file_sink::~file_sink() {}

work_return_code_t file_sink::work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output)
{
    auto inbuf = static_cast<const char*>(work_input[0].items());
    auto noutput_items = work_input[0].n_items;

    int nwritten = 0;

    do_update(); // update d_fp is reqd

    if (!d_fp)
    {
        work_input[0].n_consumed = noutput_items; // drop output on the floor
        return work_return_code_t::WORK_OK;
    }
        

    while (nwritten < noutput_items) {
        const int count = fwrite(inbuf, d_itemsize, noutput_items - nwritten, d_fp);
        if (count == 0) {
            if (ferror(d_fp)) {
                std::stringstream s;
                s << "file_sink write failed with error " << fileno(d_fp) << std::endl;
                throw std::runtime_error(s.str());
            } else { // is EOF
                break;
            }
        }
        nwritten += count;
        inbuf += count * d_itemsize;
    }

    if (d_unbuffered)
        fflush(d_fp);

    work_input[0].n_consumed = nwritten;
    return work_return_code_t::WORK_OK;
}

bool file_sink::stop()
{
    do_update();
    fflush(d_fp);
    return true;
}

} // namespace fileio
} /* namespace gr */
