#pragma once

#include <gnuradio/tag.hpp>
#include <functional>
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
    std::string _type;
    uint64_t _total_read = 0;
    uint64_t _total_written = 0;

    void set_type(const std::string& type) { _type = type; }
    virtual ~buffer() {}

public:
    virtual void* read_ptr() = 0;
    virtual void* write_ptr() = 0;

    // virtual int capacity() = 0;
    // virtual int size() = 0;

    /**
     * @brief Return current read state of buffer reader
     *
     * @param info Reference to \buffer_info_t struct
     * @return true if info is valid
     * @return false if info is not valid (e.g. could not acquire mutex)
     */
    virtual bool read_info(buffer_info_t& info) = 0;

    /**
     * @brief Return current read state of buffer writer
     *
     * @param info Reference to \buffer_info_t struct
     * @return true if info is valid
     * @return false if info is not valid (e.g. could not acquire mutex)
     */
    virtual bool write_info(buffer_info_t& info) = 0;

    /**
     * @brief Return the tags associated with this buffer
     *
     * @param num_items Number of items that will be associated with the work call, and
     * thus return the tags from the current read pointer to this specified number of
     * items
     * @return std::vector<tag_t> Returns the vector of tags
     */
    virtual std::vector<tag_t> get_tags(unsigned int num_items)
    {
        return std::vector<tag_t>{};
    }; // not virtual just yet = 0;

    virtual void add_tags(unsigned int num_items,
                          std::vector<tag_t>& tags){}; // not virtual just yet = 0;

    /**
     * @brief Updates the read pointers of the buffer 
     * 
     * @param num_items 
     */
    virtual void post_read(int num_items) = 0;

    /**
     * @brief Updates the write pointers of the buffer
     * 
     * @param num_items 
     */
    virtual void post_write(int num_items) = 0;

    /**
     * @brief Copy items from another buffer into this buffer
     * 
     * Note: This is not valid for all buffers, e.g. domain adapters
     * 
     * @param from 
     * @param nitems 
     */
    virtual void copy_items(std::shared_ptr<buffer> from, int nitems) = 0;

    /**
     * @brief Set the name of the buffer
     * 
     * @param name 
     */
    void set_name(const std::string& name) { _name = name; }

    /**
     * @brief Get the name of the buffer
     * 
     * @return std::string 
     */
    std::string name() { return _name; }

    /**
     * @brief Get the type of the buffer
     * 
     * @return std::string 
     */
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


typedef std::function<std::shared_ptr<buffer>(
    size_t, size_t, std::shared_ptr<buffer_properties>)>
    buffer_factory_function;

} // namespace gr
