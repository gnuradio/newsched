#pragma once

#include <gnuradio/tag.hpp>
#include <cstdint>
#include <vector>

#include <gnuradio/buffer.hpp>

namespace gr {

/**
 * @brief Struct for passing all information needed for input data to block::work
 *
 */
struct block_work_input {
    int n_items;
    buffer_sptr buffer;
    int n_consumed; // output the number of items that were consumed on the work() call

    block_work_input(int n_items_, buffer_sptr p_buf_)
        : n_items(n_items_), buffer(p_buf_), n_consumed(-1)
    {
    }

    void* items() { return buffer->read_ptr(); }
    uint64_t nitems_read() { return buffer->total_read(); }

    void add_tag(tag_t& tag)
    {
        buffer->add_tag(tag);
    }
    void add_tag(uint64_t offset,
                 pmtf::pmt_sptr key,
                 pmtf::pmt_sptr value,
                 pmtf::pmt_sptr srcid = nullptr)
    {
        buffer->add_tag(offset, key, value, srcid);
    }
};

/**
 * @brief Struct for passing all information needed for output data from block::work
 *
 */
struct block_work_output {
    int n_items;
    buffer_sptr buffer;
    int n_produced; // output the number of items that were consumed on the work() call

    block_work_output(int _n_items, buffer_sptr p_buf_)
        : n_items(_n_items), buffer(p_buf_), n_produced(-1)
    {
    }

    void* items() { return buffer->write_ptr(); }
    uint64_t nitems_written() { return buffer->total_written(); }

    void add_tag(tag_t& tag)
    {
        buffer->add_tag(tag);
    }
    void add_tag(uint64_t offset,
                 pmtf::pmt_sptr key,
                 pmtf::pmt_sptr value,
                 pmtf::pmt_sptr srcid = nullptr)
    {
        buffer->add_tag(offset, key, value, srcid);
    }
};

/**
 * @brief Enum for return codes from calls to block::work
 *
 */
enum class work_return_code_t {
    WORK_ERROR = -100, /// error occurred in the work function
    WORK_INSUFFICIENT_OUTPUT_ITEMS =
        -3, /// work requires a larger output buffer to produce output
    WORK_INSUFFICIENT_INPUT_ITEMS =
        -2, /// work requires a larger input buffer to produce output
    WORK_DONE =
        -1, /// this block has completed its processing and the flowgraph should be done
    WORK_OK = 0, /// work call was successful and return values in i/o structs are valid
};

} // namespace gr
