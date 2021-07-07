#pragma once

#include <gnuradio/tag.hh>
#include <cstdint>
#include <vector>

#include <gnuradio/buffer.hh>
#include <gnuradio/buffer_cpu_simple.hh>

namespace gr {

/**
 * @brief Struct for passing all information needed for input data to block::work
 *
 */
class block_work_input
{
private:
    int n_items;
    buffer_reader_sptr buffer;
    int n_consumed; // output the number of items that were consumed on the work() call

public:
    block_work_input(int n_items_, buffer_reader_sptr p_buf_)
        : n_items(n_items_), buffer(p_buf_), n_consumed(-1){};

    block_work_input(int n_items_, int item_size, void* buf)
        : n_items(n_items_), n_consumed(-1)
    {
        auto inbuf_props = std::make_shared<buffer_properties>();
        inbuf_props->set_buffer_size(2 * n_items);
        auto inbuf = std::make_shared<buffer_cpu_simple>(n_items, item_size, inbuf_props);
        std::memcpy(inbuf->read_ptr(0), buf, item_size * n_items);
        buffer = std::make_shared<buffer_cpu_simple_reader>(inbuf, inbuf_props);
    };

    static std::shared_ptr<block_work_input> make(int num_items, int item_size, void* buf)
    {
        return std::make_shared<block_work_input>(num_items, item_size, buf);
    };

    void* items() { return buffer->read_ptr(); }
    uint64_t nitems_read() { return buffer->total_read(); }
    void consume(int num) { n_consumed = num; }

    std::vector<tag_t> tags_in_window(const uint64_t item_start, const uint64_t item_end)
    {
        return buffer->tags_in_window(item_start, item_end);
    }
};

/**
 * @brief Struct for passing all information needed for output data from block::work
 *
 */
class block_work_output
{
public:
    block_work_output(int _n_items, buffer_sptr p_buf_)
        : n_items(_n_items), n_produced(-1), buffer(p_buf_){};

    block_work_output(int n_items_, int item_size, void* buf)
        : n_items(n_items_), n_produced(-1)
    {
        auto outbuf_props = std::make_shared<buffer_properties>();
        outbuf_props->set_buffer_size(n_items);
        buffer = std::make_shared<buffer_cpu_simple>(n_items, item_size, outbuf_props);
    };

    static std::shared_ptr<block_work_output>
    make(int num_items, int item_size, void* buf)
    {
        return std::make_shared<block_work_output>(num_items, item_size, buf);
    };


    void* items() { return buffer->write_ptr(); }
    uint64_t nitems_written() { return buffer->total_written(); }
    void produce(int num) { n_produced = num; }

    void add_tag(tag_t& tag) { buffer->add_tag(tag); }
    void add_tag(uint64_t offset,
                 pmtf::pmt_sptr key,
                 pmtf::pmt_sptr value,
                 pmtf::pmt_sptr srcid = nullptr)
    {
        buffer->add_tag(offset, key, value, srcid);
    }

    int n_items;
    int n_produced; // output the number of items that were consumed on the work() call
    buffer_sptr buffer;
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
