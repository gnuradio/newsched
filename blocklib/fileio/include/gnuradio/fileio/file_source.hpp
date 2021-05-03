/* -*- c++ -*- */
/*
 * Copyright 2012, 2018 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#pragma once

#include <gnuradio/sync_block.hpp>
#include <pmt/pmtf.hpp>
#include <mutex>

namespace gr {
namespace fileio {

/*!
 * \brief Read stream from file
 * \ingroup file_operators_blk
 */
class file_source : virtual public sync_block
{
public:
    // gr::blocks::file_source::sptr
    typedef std::shared_ptr<file_source> sptr;

    /*!
     * \brief Create a file source.
     *
     * Opens \p filename as a source of items into a flowgraph. The
     * data is expected to be in binary format, item after item. The
     * \p itemsize of the block determines the conversion from bits
     * to items. The first \p offset items (default 0) will be
     * skipped.
     *
     * If \p repeat is turned on, the file will repeat the file after
     * it's reached the end.
     *
     * If \p len is non-zero, only items (offset, offset+len) will
     * be produced.
     *
     * \param itemsize        the size of each item in the file, in bytes
     * \param filename        name of the file to source from
     * \param repeat  repeat file from start
     * \param offset  begin this many items into file
     * \param len     produce only items (offset, offset+len)
     */
    static sptr make(size_t itemsize,
                     const char* filename,
                     bool repeat = false,
                     uint64_t offset = 0,
                     uint64_t len = 0);

    /*!
     * \brief seek file to \p seek_point relative to \p whence
     *
     * \param seek_point      sample offset in file
     * \param whence  one of SEEK_SET, SEEK_CUR, SEEK_END (man fseek)
     */
    bool seek(int64_t seek_point, int whence);

    /*!
     * \brief Opens a new file.
     *
     * \param filename        name of the file to source from
     * \param repeat  repeat file from start
     * \param offset  begin this many items into file
     * \param len     produce only items [offset, offset+len)
     */
    void open(const char* filename, bool repeat, uint64_t offset = 0, uint64_t len = 0);

    /*!
     * \brief Close the file handle.
     */
    void close();

    /*!
     * \brief Add a stream tag to the first sample of the file if true
     */
    void set_begin_tag(pmtf::pmt_sptr val);

private:
    const size_t d_itemsize;
    uint64_t d_start_offset_items;
    uint64_t d_length_items;
    uint64_t d_items_remaining;
    FILE* d_fp;
    FILE* d_new_fp;
    bool d_repeat;
    bool d_updated;
    bool d_file_begin;
    bool d_seekable;
    long d_repeat_cnt;
    pmtf::pmt_sptr d_add_begin_tag;

    std::mutex fp_mutex;
    pmtf::pmt_sptr _id;

    void do_update();

public:
    file_source(size_t itemsize,
                     const char* filename,
                     bool repeat,
                     uint64_t offset,
                     uint64_t len);
    ~file_source() override;


    work_return_code_t work(std::vector<block_work_input>& work_input,
                            std::vector<block_work_output>& work_output) override;
};

} /* namespace blocks */
} /* namespace gr */
