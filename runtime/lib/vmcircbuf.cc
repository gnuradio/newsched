#include <gnuradio/vmcircbuf.hh>

#include "vmcircbuf_mmap_shm_open.hh"
#include "vmcircbuf_sysv_shm.hh"
#include <cstring>
#include <mutex>
#include <numeric>
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

vmcirc_buffer::vmcirc_buffer(size_t num_items,
                             size_t item_size,
                             size_t granularity,
                             std::shared_ptr<buffer_properties> buf_properties)
    : buffer(num_items, item_size, buf_properties)
{
    // This is the code from gnuradio that forces buffers to align with items

    auto min_buffer_items = granularity / std::gcd(item_size, granularity);
    if (num_items % min_buffer_items != 0)
        num_items = ((num_items / min_buffer_items) + 1) * min_buffer_items;

    // Add warning

    // Ensure that the instantiated buffer is a multiple of the granularity
    auto requested_size = num_items * item_size;
    auto npages = requested_size / granularity;
    if (requested_size != granularity * npages) {
        npages++;
    }
    auto actual_size = granularity * npages;

    // _num_items = num_items;
    _num_items = actual_size / item_size;
    _item_size = item_size;
    // _buf_size = _num_items * _item_size;
    _buf_size = actual_size;
    _write_index = 0;
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
