#pragma once

#include <gnuradio/tag.hpp>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
namespace gr {

/**
 * @brief Information about the current state of the buffer
 *
 * The buffer_info_t class is used to return information about the current state of the
 * buffer for reading or writing, as in how many items are contained, or how much space is
 * there to write into, as well as the total items read/written
 *
 */
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

    unsigned int _read_index;
    unsigned int _write_index;
    unsigned int _num_items;
    unsigned int _item_size;
    unsigned int _buf_size;

    void set_type(const std::string& type) { _type = type; }
    virtual ~buffer() {}

    std::mutex _buf_mutex;
    std::vector<tag_t> _tags;

public:
    unsigned int size() { return _buf_size; }
    unsigned int num_items() { return _num_items; }
    unsigned int item_size() { return _item_size; }
    virtual void* read_ptr() = 0;
    virtual void* write_ptr() = 0;

    /**
     * @brief Return current buffer state for reading
     *
     * @param info Reference to \buffer_info_t struct
     * @return true if info is valid
     * @return false if info is not valid (e.g. could not acquire mutex)
     */
    virtual bool read_info(buffer_info_t& info) = 0;

    /**
     * @brief Return current buffer state for writing
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
    virtual std::vector<tag_t> get_tags(unsigned int num_items);


    virtual void add_tags(unsigned int num_items, std::vector<tag_t>& tags);

    virtual const std::vector<tag_t>& tags();

    virtual std::vector<tag_t> tags_in_window(const uint64_t item_start, const uint64_t item_end);
    virtual void add_tag(tag_t tag);
    virtual void add_tag(uint64_t offset,
                 pmtf::pmt_sptr key,
                 pmtf::pmt_sptr value,
                 pmtf::pmt_sptr srcid = nullptr);
    void propagate_tags(std::shared_ptr<buffer> p_in_buf, int n_consumed);

    void prune_tags(int n_consumed);

    /**
     * @brief Updates the read pointers of the buffer
     *
     * @param num_items Number of items that were read from the buffer
     */
    virtual void post_read(int num_items) = 0;

    /**
     * @brief Updates the write pointers of the buffer
     *
     * @param num_items Number of items that were written to the buffer
     */
    virtual void post_write(int num_items) = 0;

    /**
     * @brief Copy items from another buffer into this buffer
     *
     * Note: This is not valid for all buffers, e.g. domain adapters
     *
     * @param from The other buffer that needs to be copied into this buffer
     * @param nitems The number of items to copy
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

    uint64_t total_written() const { return _total_written; }
    uint64_t total_read() const { return _total_read; }
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
    buffer_properties() {}
    virtual ~buffer_properties() {}
};

typedef std::function<std::shared_ptr<buffer>(
    size_t, size_t, std::shared_ptr<buffer_properties>)>
    buffer_factory_function;

} // namespace gr
