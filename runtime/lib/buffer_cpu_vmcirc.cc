#include <gnuradio/buffer_cpu_vmcirc.hh>
#include "buffer_cpu_vmcirc_mmap_shm.hh"
#include "buffer_cpu_vmcirc_sysv_shm.hh"
#include <cstring>
#include <mutex>

// Doubly mapped circular buffer class

namespace gr {

std::mutex s_vm_mutex;

buffer_sptr buffer_cpu_vmcirc::make(size_t num_items,
                                size_t item_size,
                                std::shared_ptr<buffer_properties> buffer_properties)
{
    auto bp = std::static_pointer_cast<buffer_cpu_vmcirc_properties>(buffer_properties);
    if (bp != nullptr) {
        switch (bp->buffer_type()) {
        case buffer_cpu_vmcirc_type::AUTO:
        case buffer_cpu_vmcirc_type::SYSV_SHM:
            return buffer_sptr(new buffer_cpu_vmcirc_sysv_shm(num_items, item_size, buffer_properties));
        case buffer_cpu_vmcirc_type::MMAP_SHM:
            return buffer_sptr(new buffer_cpu_vmcirc_mmap_shm(num_items, item_size, buffer_properties));
        default:
            throw std::runtime_error("Invalid vmcircbuf buffer_type");
        }

    } else {
        throw std::runtime_error(
            "Failed to cast buffer properties to vmcirc_buffer_properties");
    }
}

buffer_cpu_vmcirc::buffer_cpu_vmcirc(size_t num_items, size_t item_size, std::shared_ptr<buffer_properties> buf_properties)
    : buffer(num_items, item_size, buf_properties)
{
}

void* buffer_cpu_vmcirc::write_ptr() { return (void*)&_buffer[_write_index]; }

void buffer_cpu_vmcirc_reader::post_read(int num_items)
{
    std::scoped_lock guard(_rdr_mutex);

    // advance the read pointer
    _read_index += num_items * _buffer->item_size();
    if (_read_index >= _buffer->buf_size()) {
        _read_index -= _buffer->buf_size();
    }
    _total_read += num_items;
}
void buffer_cpu_vmcirc::post_write(int num_items)
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

std::shared_ptr<buffer_reader> buffer_cpu_vmcirc::add_reader(std::shared_ptr<buffer_properties> buf_props)
{
    std::shared_ptr<buffer_cpu_vmcirc_reader> r(
        new buffer_cpu_vmcirc_reader(shared_from_this(), buf_props, _write_index));
    _readers.push_back(r.get());
    return r;
}

} // namespace gr
