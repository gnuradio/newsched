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

    void set_type(const std::string& type) { _type = type; }
    virtual ~buffer() {}

    std::mutex _buf_mutex;
    std::vector<tag_t> _tags;

public:
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
    virtual std::vector<tag_t> get_tags(unsigned int num_items)
    {
        std::scoped_lock guard(_buf_mutex);

        // Find all the tags from total_read to total_read+offset
        std::vector<tag_t> ret;
        for (auto& tag : _tags) {
            if (tag.offset >= _total_read && tag.offset < _total_read + num_items) {
                ret.push_back(tag);
            }
        }

        return ret;
    }


    virtual void add_tags(unsigned int num_items, std::vector<tag_t>& tags)
    {
        std::scoped_lock guard(_buf_mutex);

        for (auto tag : tags) {
            if (tag.offset < _total_written - num_items || tag.offset >= _total_written) {

            } else {
                _tags.push_back(tag);
            }
        }
    }

    const std::vector<tag_t>& tags() const { return _tags; }
    void add_tag(tag_t tag)
    {
        std::scoped_lock guard(_buf_mutex);
        _tags.push_back(tag);
    }
    void add_tag(uint64_t offset,
                 pmtf::pmt_sptr key,
                 pmtf::pmt_sptr value,
                 pmtf::pmt_sptr srcid = nullptr)
    {
        std::scoped_lock guard(_buf_mutex);
        _tags.emplace_back(offset, key, value, srcid);
    }

    void propagate_tags(std::shared_ptr<buffer> p_in_buf, int n_consumed)
    {
        std::scoped_lock guard(_buf_mutex);
        for (auto& t : p_in_buf->tags()) {
            // Propagate the tags that occurred in the processed window
            if (t.offset >= p_in_buf->total_read() &&
                t.offset < p_in_buf->total_read() + n_consumed) {
                // std::cout << "adding tag" << std::endl;
                _tags.push_back(t);
            }
        }
    }

    void prune_tags(int n_consumed)
    {
        std::scoped_lock guard(_buf_mutex);
        auto t = std::begin(_tags);
        while (t != std::end(_tags)) {
            // Do some stuff
            if (t->offset < total_read() + n_consumed) {
                t = _tags.erase(t);
                // std::cout << "removing tag" << std::endl;
            } else {
                ++t;
            }
        }
    }

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
