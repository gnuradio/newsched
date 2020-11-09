#pragma once

#include <gnuradio/tag.hpp>
#include <functional>
#include <memory>
#include <vector>

namespace gr {


/**
 * @brief Buffer position defines where the buffer sits relative to the sub-graph
 *
 * If it is as the edge, it should be INGRESS or EGRESS, if that is a necessary
 * consideration for the domain scheduler
 */
enum class buffer_position_t { NORMAL, INGRESS, EGRESS };


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
    std::string _type;
    int _total_read = 0;
    int _total_written = 0;

    void set_type(const std::string& type) { _type = type; }
    
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

    std::string type() { return _type; }
};

typedef std::shared_ptr<buffer> buffer_sptr;

/**
 * @brief Base class for passing custom buffer properties into factory method
 *
 * Buffer Properties will vary according to the particular buffer
 */
class buffer_properties
{
public:
    // typedef sptr std::shared_ptr<buffer_properties>;
    buffer_properties() {}
    virtual ~buffer_properties() {}
    // buffer_factory_function bff() { return _bff; }

// private:
    // buffer_factory_function _bff;
};


typedef std::function<std::shared_ptr<buffer>(size_t, size_t, std::shared_ptr<buffer_properties>)>
    buffer_factory_function;

} // namespace gr
