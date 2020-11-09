#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
// #include <boost/thread/mutex.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <gnuradio/cudabuffer_pinned.hpp>


// typedef boost::unique_lock<boost::mutex> scoped_lock;

namespace gr {
cuda_buffer_pinned::cuda_buffer_pinned(size_t num_items, size_t item_size)
    : _num_items(num_items),
      _item_size(item_size),
      _buf_size(_num_items * _item_size),
      _read_index(0),
      _write_index(0)
{
    if (!cudaHostAlloc((void **)&_pinned_buffer, _buf_size * 2, 0) == cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA pinned memory");
    }
}
cuda_buffer_pinned::~cuda_buffer_pinned() { cudaFree(_pinned_buffer); }

buffer_sptr cuda_buffer_pinned::make(size_t num_items,
                             size_t item_size,
                             std::shared_ptr<buffer_properties> buffer_properties)
{
    return buffer_sptr(new cuda_buffer_pinned(num_items, item_size));
}

int cuda_buffer_pinned::size()
{ // in number of items
    int w = _write_index;
    int r = _read_index;

    if (w < r)
        w += _buf_size;
    return (w - r) / _item_size;
}
int cuda_buffer_pinned::capacity() { return _num_items; }

void* cuda_buffer_pinned::read_ptr() { return (void*)&_pinned_buffer[_read_index]; }
void* cuda_buffer_pinned::write_ptr() { return (void*)&_pinned_buffer[_write_index]; }

bool cuda_buffer_pinned::read_info(buffer_info_t& info)
{
    // Need to lock the buffer to freeze the current state
    // if (!_buf_mutex.try_lock()) {
    //     return false;
    // }
    // _buf_mutex.lock();
    std::lock_guard<std::mutex> guard(_buf_mutex);

    info.ptr = read_ptr();
    info.n_items = size();
    info.item_size = _item_size;

    return true;
}

bool cuda_buffer_pinned::write_info(buffer_info_t& info)
{
    // if (!_buf_mutex.try_lock()) {
    //     return false;
    // }
    // _buf_mutex.lock();
    std::lock_guard<std::mutex> guard(_buf_mutex);

    info.ptr = write_ptr();
    info.n_items =
        capacity() - size() - 1; // always keep the write pointer 1 behind the read ptr
    if (info.n_items < 0)
        info.n_items = 0;
    info.item_size = _item_size;

    return true;
}

void cuda_buffer_pinned::cancel() {}
// { _buf_mutex.unlock(); }

void cuda_buffer_pinned::post_read(int num_items)
{
    std::lock_guard<std::mutex> guard(_buf_mutex);
    // advance the read pointer
    _read_index += num_items * _item_size;
    if (_read_index >= _buf_size) {
        _read_index -= _buf_size;
    }
    // _buf_mutex.unlock();
}
void cuda_buffer_pinned::post_write(int num_items)
{
    std::lock_guard<std::mutex> guard(_buf_mutex);

    unsigned int bytes_written = num_items * _item_size;
    int wi1 = _write_index;
    int wi2 = _write_index + _buf_size;
    // num_items were written to the buffer
    // copy the data to the second half of the buffer

    int num_bytes_1 = std::min(_buf_size - wi1, bytes_written);
    int num_bytes_2 = bytes_written - num_bytes_1;

    memcpy(&_pinned_buffer[wi2], &_pinned_buffer[wi1], num_bytes_1);
    if (num_bytes_2)
        memcpy(&_pinned_buffer[0], &_pinned_buffer[_buf_size], num_bytes_2);

    // advance the write pointer
    _write_index += bytes_written;
    if (_write_index >= _buf_size) {
        _write_index -= _buf_size;
    }

    // _buf_mutex.unlock();
}

void cuda_buffer_pinned::copy_items(std::shared_ptr<buffer> from, int nitems)
{
    std::lock_guard<std::mutex> guard(_buf_mutex);
    memcpy(write_ptr(), from->write_ptr(), nitems * _item_size);
}
} // namespace gr