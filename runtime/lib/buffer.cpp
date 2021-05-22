#include <gnuradio/buffer.hpp>

namespace gr {
bool buffer::write_info(buffer_info_t& info)
{
    std::scoped_lock guard(_buf_mutex);

    // Find the max number of items available across readers
    uint64_t n_available = 0;
    for (auto& r : _readers) {
        auto n = r->items_available();
        if (n > n_available) {
            n_available = n;
        }
    }

    info.ptr = write_ptr();
    info.n_items = _num_items - n_available -
                   1; // always keep the write pointer 1 behind the read ptr
    if (info.n_items < 0)
        info.n_items = 0;
    info.n_items =
        std::min(info.n_items, (int)(_num_items / 2)); // move to a max_fill parameter
    info.item_size = _item_size;
    info.total_items = _total_written;

    return true;
}
void buffer::add_tags(size_t num_items, std::vector<tag_t>& tags)
{
    std::scoped_lock guard(_buf_mutex);

    for (auto tag : tags) {
        if (tag.offset < _total_written - num_items || tag.offset >= _total_written) {

        } else {
            _tags.push_back(tag);
        }
    }
}

std::vector<tag_t> buffer_reader::tags_in_window(const uint64_t item_start,
                                          const uint64_t item_end)
{
    std::scoped_lock guard(*(_buffer->mutex()));

    std::vector<tag_t> ret;
    for (auto& t : _buffer->tags()) {
        if (t.offset >= total_read() + item_start && t.offset < total_read() + item_end) {
            ret.push_back(t);
        }
    }
    return ret;
}

void buffer::add_tag(tag_t tag)
{
    std::scoped_lock guard(_buf_mutex);
    _tags.push_back(tag);
}
void buffer::add_tag(uint64_t offset,
                     pmtf::pmt_wrap key,
                     pmtf::pmt_wrap value,
                     pmtf::pmt_wrap srcid)
{
    std::scoped_lock guard(_buf_mutex);
    _tags.emplace_back(offset, key, value, srcid);
}

void buffer::propagate_tags(std::shared_ptr<buffer_reader> p_in_buf, int n_consumed)
{
    std::scoped_lock guard(_buf_mutex);
    for (auto& t : p_in_buf->tags()) {
        // Propagate the tags that occurred in the processed window
        if (t.offset >= total_written() && t.offset < total_written() + n_consumed) {
            // std::cout << "adding tag" << std::endl;
            _tags.push_back(t);
        }
    }
}

void buffer::prune_tags()
{
    std::scoped_lock guard(_buf_mutex);

    // Find the min number of items available across readers
    auto n_read = total_written();
    for (auto& r : _readers) {
        auto n = r->total_read();
        if (n < n_read) {
            n_read = n;
        }
    }

    auto t = std::begin(_tags);
    while (t != std::end(_tags)) {
        // Do some stuff
        if (t->offset < n_read) {
            t = _tags.erase(t);
            // std::cout << "removing tag" << std::endl;
        } else {
            ++t;
        }
    }
}

    size_t buffer_reader::items_available()
    {
        size_t w = _buffer->write_index();
        size_t r = _read_index;

        if (w < r)
            w += _buffer->buf_size();
        return (w - r) / _buffer->item_size();
    }

    bool buffer_reader::read_info(buffer_info_t& info)
    {
        // std::scoped_lock guard(_rdr_mutex);

        info.ptr = _buffer->read_ptr(_read_index);
        info.n_items = items_available();
        info.item_size = _buffer->item_size();
        info.total_items = _total_read;

        return true;
    }

    /**
     * @brief Return the tags associated with this buffer
     *
     * @param num_items Number of items that will be associated with the work call, and
     * thus return the tags from the current read pointer to this specified number of
     * items
     * @return std::vector<tag_t> Returns the vector of tags
     */
    std::vector<tag_t> buffer_reader::get_tags(size_t num_items)
    {
        std::scoped_lock guard(*(_buffer->mutex()));

        // Find all the tags from total_read to total_read+offset
        std::vector<tag_t> ret;
        for (auto& tag : _buffer->tags()) {
            if (tag.offset >= total_read() && tag.offset < total_read() + num_items) {
                ret.push_back(tag);
            }
        }

        return ret;
    }

    const std::vector<tag_t>& buffer_reader::tags() const
    {
        std::scoped_lock guard(*(_buffer->mutex()));
        return _buffer->tags();
    }


} // namespace gr
