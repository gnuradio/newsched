#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <gnuradio/cudabuffer_pinned.hpp>

namespace gr {
cuda_buffer_pinned::cuda_buffer_pinned(size_t num_items, size_t item_size, std::shared_ptr<buffer_properties> buf_properties)
    : buffer(num_items, item_size, buf_properties)
{
    if (!cudaHostAlloc((void**)&_pinned_buffer, _buf_size * 2, 0) == cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA pinned memory");
    }
}
cuda_buffer_pinned::~cuda_buffer_pinned() { cudaFree(_pinned_buffer); }

buffer_sptr cuda_buffer_pinned::make(size_t num_items,
                                     size_t item_size,
                                     std::shared_ptr<buffer_properties> buffer_properties)
{
    return buffer_sptr(new cuda_buffer_pinned(num_items, item_size, buffer_properties));
}

void* cuda_buffer_pinned::read_ptr(size_t index) { return (void*)&_pinned_buffer[index]; }
void* cuda_buffer_pinned::write_ptr() { return (void*)&_pinned_buffer[_write_index]; }

void cuda_buffer_pinned_reader::post_read(int num_items)
{
    std::lock_guard<std::mutex> guard(_rdr_mutex);
    // advance the read pointer
    _read_index += num_items * _buffer->item_size();
    if (_read_index >= _buffer->buf_size()) {
        _read_index -= _buffer->buf_size();
    }
    _total_read += num_items;
}

void cuda_buffer_pinned::post_write(int num_items)
{
    std::lock_guard<std::mutex> guard(_buf_mutex);

    size_t bytes_written = num_items * _item_size;
    size_t wi1 = _write_index;
    size_t wi2 = _write_index + _buf_size;
    // num_items were written to the buffer
    // copy the data to the second half of the buffer

    size_t num_bytes_1 = std::min(_buf_size - wi1, bytes_written);
    size_t num_bytes_2 = bytes_written - num_bytes_1;

    memcpy(&_pinned_buffer[wi2], &_pinned_buffer[wi1], num_bytes_1);
    if (num_bytes_2)
        memcpy(&_pinned_buffer[0], &_pinned_buffer[_buf_size], num_bytes_2);

    // advance the write pointer
    _write_index += bytes_written;
    if (_write_index >= _buf_size) {
        _write_index -= _buf_size;
    }
}

std::shared_ptr<buffer_reader> cuda_buffer_pinned::add_reader(std::shared_ptr<buffer_properties> buf_props)
{
    std::shared_ptr<cuda_buffer_pinned_reader> r(
        new cuda_buffer_pinned_reader(shared_from_this(), buf_props, _write_index));
    _readers.push_back(r.get());
    return r;
}

} // namespace gr