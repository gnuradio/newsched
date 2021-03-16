#include <gnuradio/buffer.hpp>

namespace gr {

std::vector<tag_t> buffer::get_tags(unsigned int num_items)
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


void buffer::add_tags(unsigned int num_items, std::vector<tag_t>& tags)
{
    std::scoped_lock guard(_buf_mutex);

    for (auto tag : tags) {
        if (tag.offset < _total_written - num_items || tag.offset >= _total_written) {

        } else {
            _tags.push_back(tag);
        }
    }
}

const std::vector<tag_t>& buffer::tags() { return _tags; }

std::vector<tag_t> buffer::tags_in_window(const uint64_t item_start,
                                          const uint64_t item_end)
{
    std::scoped_lock guard(_buf_mutex);
    std::vector<tag_t> ret;
    for (auto& t : _tags) {
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
                     pmtf::pmt_sptr key,
                     pmtf::pmt_sptr value,
                     pmtf::pmt_sptr srcid)
{
    std::scoped_lock guard(_buf_mutex);
    _tags.emplace_back(offset, key, value, srcid);
}

void buffer::propagate_tags(std::shared_ptr<buffer> p_in_buf, int n_consumed)
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

void buffer::prune_tags(int n_consumed)
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
} // namespace gr
