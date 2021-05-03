#include <gnuradio/vmcircbuf.hpp>

#include "vmcircbuf_mmap_shm_open.hpp"
#include "vmcircbuf_sysv_shm.hpp"
#include <cstring>
#include <mutex>

// Doubly mapped circular buffer class

namespace gr {

std::mutex s_vm_mutex;

buffer_sptr vmcirc_buffer::make(size_t num_items,
                                size_t item_size,
                                std::shared_ptr<buffer_properties> buffer_properties)
{
    auto bp = std::static_pointer_cast<vmcirc_buffer_properties>(buffer_properties);
    if (bp != nullptr) {
        switch (bp->buffer_type()) {
        case vmcirc_buffer_type::AUTO:
        case vmcirc_buffer_type::SYSV_SHM:
            return buffer_sptr(new vmcircbuf_sysv_shm(num_items, item_size, buffer_properties));
        case vmcirc_buffer_type::MMAP_SHM:
            return buffer_sptr(new vmcircbuf_mmap_shm_open(num_items, item_size, buffer_properties));
        default:
            throw std::runtime_error("Invalid vmcircbuf buffer_type");
        }

    } else {
        throw std::runtime_error(
            "Failed to cast buffer properties to vmcirc_buffer_properties");
    }
}

vmcirc_buffer::vmcirc_buffer(size_t num_items, size_t item_size, std::shared_ptr<buffer_properties> buf_properties)
    : buffer(num_items, item_size, buf_properties)
{
}

void* vmcirc_buffer::write_ptr() { return (void*)&_buffer[_write_index]; }

void vmcirc_buffer_reader::post_read(int num_items)
{
    std::scoped_lock guard(_rdr_mutex);

    // advance the read pointer
    _read_index += num_items * _buffer->item_size();
    if (_read_index >= _buffer->buf_size()) {
        _read_index -= _buffer->buf_size();
    }
    _total_read += num_items;
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

std::shared_ptr<buffer_reader> vmcirc_buffer::add_reader(std::shared_ptr<buffer_properties> buf_props)
{
    std::shared_ptr<vmcirc_buffer_reader> r(
        new vmcirc_buffer_reader(shared_from_this(), buf_props, _write_index));
    _readers.push_back(r.get());
    return r;
}

} // namespace gr
