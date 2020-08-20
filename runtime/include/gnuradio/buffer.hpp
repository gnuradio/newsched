#pragma once

#include <gnuradio/tag.hpp>
#include <memory>
#include <vector>

namespace gr {

struct buffer_info_t {
    void* ptr;
    int n_items; // number of items available to be read or written
    size_t item_size;
    int total_items; // the total number of items read/written from/to this buffer
};

/**
 * @brief Abstract buffer class
 *
 */
class buffer
{
protected:
    std::string _name;
    int _total_read = 0;
    int _total_written = 0;


public:
    virtual void* read_ptr() = 0;
    virtual void* write_ptr() = 0;

    // virtual int capacity() = 0;
    // virtual int size() = 0;

    virtual bool read_info(buffer_info_t& info) = 0;
    virtual bool write_info(buffer_info_t& info) = 0;
    virtual void cancel() = 0;

    virtual std::vector<tag_t> get_tags(unsigned int num_items)
    {
        return std::vector<tag_t>{};
    }; // not virtual just yet = 0;
    virtual void add_tags(unsigned int num_items,
                          std::vector<tag_t>& tags){}; // not virtual just yet = 0;

    virtual void post_read(int num_items) = 0;
    virtual void post_write(int num_items) = 0;

    // This is not valid for all buffers, e.g. domain adapters
    virtual void copy_items(std::shared_ptr<buffer> from, int nitems) = 0;

    void set_name(const std::string& name) { _name = name; }
    std::string name() { return _name; }
};

typedef std::shared_ptr<buffer> buffer_sptr;

} // namespace gr