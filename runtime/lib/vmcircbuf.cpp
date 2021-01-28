#include <gnuradio/vmcircbuf.hpp>

#include "vmcircbuf_sysv_shm.hpp"
#include "vmcircbuf_mmap_shm_open.hpp"
#include <mutex>
#include <cstring>

// Doubly mapped circular buffer class
// For now, just do this as the sysv_shm flavor
// expand out with the preferences and the factories later

namespace gr {

std::mutex s_vm_mutex;

buffer_sptr vmcirc_buffer::make(size_t num_items,
                                size_t item_size,
                                std::shared_ptr<buffer_properties> buffer_properties)
{
    auto bp = std::dynamic_pointer_cast<vmcirc_buffer_properties>(buffer_properties);
    if (bp != nullptr) {
        switch (bp->buffer_type()) {
        case vmcirc_buffer_type::AUTO:
        case vmcirc_buffer_type::SYSV_SHM:
            return buffer_sptr(new vmcircbuf_sysv_shm(num_items, item_size));
        case vmcirc_buffer_type::MMAP_SHM:
            return buffer_sptr(new vmcircbuf_mmap_shm_open(num_items, item_size));
        }

    } else {
        throw std::runtime_error(
            "Failed to cast buffer properties to vmcirc_buffer_properties");
    }
}

vmcirc_buffer::vmcirc_buffer(size_t num_items, size_t item_size)
{
    _num_items = num_items;
    _item_size = item_size;
    _buf_size = _num_items * _item_size;
    _read_index = 0;
    _write_index = 0;
}


int vmcirc_buffer::size()
{ // in number of items
    int w = _write_index;
    int r = _read_index;

    if (w < r)
        w += _buf_size;
    return (w - r) / _item_size;
}
int vmcirc_buffer::capacity() { return _num_items; }

void* vmcirc_buffer::read_ptr() { return (void*)&_buffer[_read_index]; }
void* vmcirc_buffer::write_ptr() { return (void*)&_buffer[_write_index]; }

bool vmcirc_buffer::read_info(buffer_info_t& info)
{
    std::scoped_lock guard(_buf_mutex);

    info.ptr = read_ptr();
    info.n_items = size();
    info.item_size = _item_size;
    info.total_items = _total_read;

    return true;
}

bool vmcirc_buffer::write_info(buffer_info_t& info)
{
    std::scoped_lock guard(_buf_mutex);

    info.ptr = write_ptr();
    info.n_items = capacity() - size() - 1;
    // always keep the write pointer 1 behind the read ptr
    // only fill the buffer half way (this should really be a scheduler not a buffer
    // decision FIXME)
    info.n_items = std::min(info.n_items, capacity() / 2);

    if (info.n_items < 0)
        info.n_items = 0;
    info.item_size = _item_size;
    info.total_items = _total_written;

    return true;
}

void vmcirc_buffer::post_read(int num_items)
{
    std::scoped_lock guard(_buf_mutex);

    // advance the read pointer
    _read_index += num_items * _item_size;
    if (_read_index >= _buf_size) {
        _read_index -= _buf_size;
    }
    _total_read += num_items;
    // _buf_mutex.unlock();
}
void vmcirc_buffer::post_write(int num_items)
{
    std::scoped_lock guard(_buf_mutex);

    unsigned int bytes_written = num_items * _item_size;

    // advance the write pointer
    _write_index += bytes_written;
    if (_write_index >= _buf_size) {
        _write_index -= _buf_size;
    }

    _total_written += num_items;
}

void vmcirc_buffer::copy_items(std::shared_ptr<buffer> from, int nitems)
{
    std::scoped_lock guard(_buf_mutex);

    memcpy(write_ptr(), from->write_ptr(), nitems * _item_size);
}

std::vector<tag_t> vmcirc_buffer::get_tags(unsigned int num_items)
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
void vmcirc_buffer::add_tags(
    unsigned int num_items,
    std::vector<tag_t>& tags) // overload with convenience functions later
{
    std::scoped_lock guard(_buf_mutex);

    for (auto tag : tags) {
        if (tag.offset < _total_written || tag.offset >= _total_written + _num_items) {

        } else {
            _tags.push_back(tag);
        }
    }
}

} // namespace gr
