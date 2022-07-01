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
    block_work_input(int n_items_, buffer_reader_sptr p_buf_, port_sptr p = nullptr)
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

    block_work_output(int _n_items, buffer_sptr p_buf_, port_sptr p = nullptr)
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
template <typename T>
class io_vec_wrap
{
private:
    std::vector<T> _vec;
    std::map<std::string, T*> _map;

public:
    T& operator[](size_t idx) { return _vec[idx]; }
    T& operator[](const std::string& name) { return *_map[name]; }
    auto begin() { return _vec.begin(); }
    auto end() { return _vec.end(); }
    auto back() { return _vec.back(); }
    auto size() { return _vec.size(); }
    auto empty() { return _vec.empty(); }
    auto clear()
    {
        _vec.clear();
        _map.clear();
    }
    void push_back(const T& element) { _vec.push_back(element); }
    void add_to_map(size_t index, const std::string& name)
    {
        _map[name] = &(_vec[index]);
    }
};

class work_io
{
public:
    work_io() {}
    work_io(const work_io&) = delete;
    work_io& operator=(const work_io&) = delete;

    io_vec_wrap<block_work_input>& inputs() { return _inputs; }
    io_vec_wrap<block_work_output>& outputs() { return _outputs; }
    void clear()
    {
        _inputs.clear();
        _outputs.clear();
    }
    void reset()
    {
        for (auto& w : _inputs) {
            w.reset();
        }
        for (auto& w : _outputs) {
            w.reset();
        }
    }
    // Convenience Methods
    void consume_each(size_t n_items)
    {
        for (auto& w : inputs()) {
            w.n_consumed = n_items;
        }
    }
    void produce_each(size_t n_items)
    {
        for (auto& w : outputs()) {
            w.n_produced = n_items;
        }
    }
    size_t min_noutput_items()
    {
        auto result = (std::min_element(
            _outputs.begin(),
            _outputs.end(),
            [](const block_work_output& lhs, const block_work_output& rhs) {
                return (lhs.n_items < rhs.n_items);
            }));
        return (*result).n_items;
    }

    size_t min_ninput_items()
    {
        auto result = (std::min_element(
            _inputs.begin(),
            _inputs.end(),
            [](const block_work_input& lhs, const block_work_input& rhs) {
                return (lhs.n_items < rhs.n_items);
            }));
        return (*result).n_items;
    }

private:
    friend class block;
    io_vec_wrap<block_work_input> _inputs;
    io_vec_wrap<block_work_output> _outputs;

    void add_input(port_sptr p)
    {
        _inputs.push_back(block_work_input(0, p->buffer_reader(), p));
        _inputs.add_to_map(_inputs.size() - 1, p->name());
    }

    void add_output(port_sptr p)
    {
        _outputs.push_back(block_work_output(0, p->buffer(), p));
        _outputs.add_to_map(_outputs.size() - 1, p->name());
    }
};

} // namespace gr
