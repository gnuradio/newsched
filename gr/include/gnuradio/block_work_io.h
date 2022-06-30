#pragma once

#include <gnuradio/buffer.h>
#include <gnuradio/port.h>
#include <gnuradio/tag.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace gr {

/**
 * @brief Struct for passing all information needed for input data to block::work
 *
 */
struct block_work_input {
    using sptr = std::shared_ptr<block_work_input>;
    size_t n_items = 0;
    buffer_reader_sptr buffer;
    size_t n_consumed =
        0; // output the number of items that were consumed on the work() call
    port_sptr port = nullptr;
    block_work_input(int n_items_, buffer_reader_sptr p_buf_, port_sptr p=nullptr)
        : n_items(n_items_), buffer(p_buf_), port(p)
    {
    }

    void reset()
    {
        n_items = 0;
        n_consumed = 0;
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
    static size_t min_n_items(const std::vector<sptr>& work_inputs)
    {
        auto result = (std::min_element(
            work_inputs.begin(), work_inputs.end(), [](const sptr& lhs, const sptr& rhs) {
                return (lhs->n_items < rhs->n_items);
            }));
        return (*result)->n_items;
    }
};

using block_work_input_sptr = block_work_input::sptr;

/**
 * @brief Struct for passing all information needed for output data from block::work
 *
 */
struct block_work_output {
    using sptr = std::shared_ptr<block_work_output>;
    size_t n_items;
    buffer_sptr buffer;
    size_t n_produced =
        0; // output the number of items that were produced on the work() call
    port_sptr port = nullptr;

    block_work_output(int _n_items, buffer_sptr p_buf_, port_sptr p=nullptr)
        : n_items(_n_items), buffer(p_buf_), port(p)
    {
    }

    void reset()
    {
        n_items = 0;
        n_produced = 0;
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
    void add_tag(uint64_t offset, tag_map map) { buffer->add_tag(offset, map); }
    void add_tag(uint64_t offset, pmtf::map map) { buffer->add_tag(offset, map); }

    static std::vector<void*> all_items(const std::vector<sptr>& work_outputs)
    {
        std::vector<void*> ret(work_outputs.size());
        for (size_t idx = 0; idx < work_outputs.size(); idx++) {
            ret[idx] = work_outputs[idx]->buffer->write_ptr();
        }

        return ret;
    }

    static size_t min_n_items(const std::vector<sptr>& work_outputs)
    {
        auto result = (std::min_element(work_outputs.begin(),
                                        work_outputs.end(),
                                        [](const sptr& lhs, const sptr& rhs) {
                                            return (lhs->n_items < rhs->n_items);
                                        }));
        return (*result)->n_items;
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


class work_io
{
public:
    friend class block;
    size_t nin() { return _inputs.size(); }
    size_t nout() { return _outputs.size(); }
    block_work_input_sptr& input(size_t idx)
    {
        return _inputs[idx];
    }
    block_work_input_sptr& input(const std::string&);
    block_work_output_sptr& output(size_t idx)
    {
        return _outputs[idx];
    }
    block_work_output_sptr& output(const std::string&);
    std::vector<block_work_input_sptr>& inputs() { return _inputs; }
    std::vector<block_work_output_sptr>& outputs() { return _outputs; }
    // block_work_input_sptr& operator[](size_t);
    // block_work_input_sptr& operator[](const std::string&);

    void consume_each(size_t n_items) {
        for (auto& w : inputs()) {
            w->n_consumed = n_items;
        }
    }
    void produce_each(size_t n_items) {
        for (auto& w : outputs()) {
            w->n_produced = n_items;
        }
    }
private:
    std::vector<block_work_input_sptr> _inputs;
    std::vector<block_work_output_sptr> _outputs;

    std::map<std::string, block_work_input_sptr> _input_name_map;
    std::map<std::string, block_work_output_sptr> _output_name_map;
};

} // namespace gr
