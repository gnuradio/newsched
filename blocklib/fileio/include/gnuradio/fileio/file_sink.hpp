/* -*- c++ -*- */
/*
 * Copyright 2004,2007,2013 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/fileio/file_sink_base.hpp>
#include <gnuradio/sync_block.hpp>

namespace gr {
namespace fileio {

/*!
 * \brief Write stream to file.
 * \ingroup file_operators_blk
 */
class file_sink : virtual public sync_block, virtual public file_sink_base
{
public:
    // gr::blocks::file_sink::sptr
    typedef std::shared_ptr<file_sink> sptr;

    /*!
     * \brief Make a file sink.
     * \param itemsize size of the input data items.
     * \param filename name of the file to open and write output to.
     * \param append if true, data is appended to the file instead of
     *        overwriting the initial content.
     */
    static sptr make(size_t itemsize, const char* filename, bool append = false);

    file_sink(size_t itemsize, const char* filename, bool append = false);
    ~file_sink() override;

    bool stop() override;

    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output) override;

private:
    const size_t d_itemsize;
};

} // namespace fileio
} // namespace gr
