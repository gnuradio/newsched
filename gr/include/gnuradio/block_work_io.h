#pragma once

#include <gnuradio/tag.h>
#include <cstdint>
#include <vector>

#include <gnuradio/buffer.h>

namespace gr {

/**
 * @brief Struct for passing all information needed for input data to block::work
 *
 */
struct block_work_input {
    using sptr = std::shared_ptr<block_work_input>;
    int n_items;
    buffer_reader_sptr buffer;
    int n_consumed =
        -1; // output the number of items that were consumed on the work() call

    block_work_input(int n_items_, buffer_reader_sptr p_buf_)
        : n_items(n_items_), buffer(p_buf_)
    {
    }

    template <typename T>
    const T* items() const
    {
        return static_cast<const T*>(buffer->read_ptr());
    }
    const void* raw_items() const { return buffer->read_ptr(); }

    uint64_t nitems_read() { return buffer->total_read(); }

    void consume(int num) { n_consumed = num; }

    std::vector<tag_t> tags_in_window(const uint64_t item_start, const uint64_t item_end)
    {
        return buffer->tags_in_window(item_start, item_end);
    }

    static std::vector<const void*> all_items(const std::vector<sptr>& work_inputs)
    {
        std::vector<const void*> ret(work_inputs.size());
        for (size_t idx = 0; idx < work_inputs.size(); idx++) {
            ret[idx] = work_inputs[idx]->buffer->read_ptr();
        }

        return ret;
    }
};

using block_work_input_sptr = block_work_input::sptr;

/**
 * @brief Struct for passing all information needed for output data from block::work
 *
 */
struct block_work_output {
    using sptr = std::shared_ptr<block_work_output>;
    int n_items;
    buffer_sptr buffer;
    int n_produced =
        -1; // output the number of items that were produced on the work() call

    block_work_output(int _n_items, buffer_sptr p_buf_)
        : n_items(_n_items), buffer(p_buf_)
    {
    }

    template <typename T>
    T* items() const
    {
        return static_cast<T*>(buffer->write_ptr());
    }
    void* raw_items() const { return buffer->write_ptr(); }

    uint64_t nitems_written() { return buffer->total_written(); }
    void produce(int num) { n_produced = num; }

    void add_tag(tag_t& tag) { buffer->add_tag(tag); }
    void
    add_tag(uint64_t offset, pmtf::pmt key, pmtf::pmt value, pmtf::pmt srcid = nullptr)
    {
        buffer->add_tag(offset, key, value, srcid);
    }

    static std::vector<void*> all_items(const std::vector<sptr>& work_outputs)
    {
        std::vector<void*> ret(work_outputs.size());
        for (size_t idx = 0; idx < work_outputs.size(); idx++) {
            ret[idx] = work_outputs[idx]->buffer->write_ptr();
        }

        return ret;
    }
};
using block_work_output_sptr = block_work_output::sptr;

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
    WORK_CALLBACK_INITIATED =
        1, /// rather than blocking in the work function, the block will call back to the
           /// parent interface when it is ready to be called again
};

} // namespace gr
