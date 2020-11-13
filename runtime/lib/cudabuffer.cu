#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
// #include <boost/thread/mutex.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <gnuradio/cudabuffer.hpp>


// typedef boost::unique_lock<boost::mutex> scoped_lock;

namespace gr {
cuda_buffer::cuda_buffer(size_t num_items, size_t item_size, cuda_buffer_type type)
    : _num_items(num_items),
      _item_size(item_size),
      _buffer_type(type),
      _buf_size(_num_items * _item_size),
      _read_index(0),
      _write_index(0),
      _type(type)
{
    _host_buffer.resize(_buf_size * 2); // double circular buffer
    cudaMalloc(
        &_device_buffer,
        _buf_size *
            2); // double circular buffer - should do something more intelligent here
    set_type("cuda_buffer_" + std::to_string((int)_type));
}
cuda_buffer::~cuda_buffer() { cudaFree(_device_buffer); }

buffer_sptr cuda_buffer::make(size_t num_items,
                              size_t item_size,
                              std::shared_ptr<buffer_properties> buffer_properties)
{
    auto cbp = std::dynamic_pointer_cast<cuda_buffer_properties>(buffer_properties);
    if (cbp != nullptr) {
        return buffer_sptr(new cuda_buffer(num_items, item_size, cbp->buffer_type()));
    } else {
        throw std::runtime_error(
            "Failed to cast buffer properties to cuda_buffer_properties");
    }
}

int cuda_buffer::size()
{ // in number of items
    int w = _write_index;
    int r = _read_index;

    if (w < r)
        w += _buf_size;
    return (w - r) / _item_size;
}
int cuda_buffer::capacity() { return _num_items; }

void* cuda_buffer::read_ptr()
{
    if (_buffer_type == cuda_buffer_type::D2H) {
        return (void*)&_host_buffer[_read_index];
    } else {
        return (void*)&_device_buffer[_read_index];
    }
}
void* cuda_buffer::write_ptr()
{
    if (_buffer_type == cuda_buffer_type::H2D) {
        return (void*)&_host_buffer[_write_index];
    } else {
        return (void*)&_device_buffer[_write_index];
    }
}

bool cuda_buffer::read_info(buffer_info_t& info)
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

bool cuda_buffer::write_info(buffer_info_t& info)
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

void cuda_buffer::post_read(int num_items)
{
    std::lock_guard<std::mutex> guard(_buf_mutex);
    // advance the read pointer
    _read_index += num_items * _item_size;
    if (_read_index >= _buf_size) {
        _read_index -= _buf_size;
    }
    // _buf_mutex.unlock();
}
void cuda_buffer::post_write(int num_items)
{
    std::lock_guard<std::mutex> guard(_buf_mutex);

    unsigned int bytes_written = num_items * _item_size;
    int wi1 = _write_index;
    int wi2 = _write_index + _buf_size;
    // num_items were written to the buffer
    // copy the data to the second half of the buffer

    int num_bytes_1 = std::min(_buf_size - wi1, bytes_written);
    int num_bytes_2 = bytes_written - num_bytes_1;

    if (_buffer_type == cuda_buffer_type::H2D) {
        cudaMemcpy(&_device_buffer[wi1],
                   &_host_buffer[wi1],
                   bytes_written,
                   cudaMemcpyHostToDevice);

        // memcpy(&_host_buffer[wi2], &_host_buffer[wi1], num_bytes_1);
        cudaMemcpy(&_device_buffer[wi2],
                   &_device_buffer[wi1],
                   num_bytes_1,
                   cudaMemcpyDeviceToDevice);
        if (num_bytes_2) {
            // memcpy(&_host_buffer[0], &_host_buffer[_buf_size], num_bytes_2);
            cudaMemcpy(&_device_buffer[0],
                       &_device_buffer[_buf_size],
                       num_bytes_2,
                       cudaMemcpyDeviceToDevice);
        }
    } else if (_buffer_type == cuda_buffer_type::D2H) {
        cudaMemcpy(&_host_buffer[wi1],
                   &_device_buffer[wi1],
                   bytes_written,
                   cudaMemcpyDeviceToHost);

        memcpy(&_host_buffer[wi2], &_host_buffer[wi1], num_bytes_1);
        // cudaMemcpy(&_device_buffer[wi2],
        //            &_device_buffer[wi1],
        //            num_bytes_1,
        //            cudaMemcpyDeviceToDevice);

        if (num_bytes_2) {
            memcpy(&_host_buffer[0], &_host_buffer[_buf_size], num_bytes_2);
            // cudaMemcpy(&_device_buffer[0],
            //            &_device_buffer[_buf_size],
            //            num_bytes_2,
            //            cudaMemcpyDeviceToDevice);
        }
    } else // D2D
    {
        cudaMemcpy(&_device_buffer[wi2],
                   &_device_buffer[wi1],
                   num_bytes_1,
                   cudaMemcpyDeviceToDevice);
        if (num_bytes_2)
            cudaMemcpy(&_device_buffer[0],
                       &_device_buffer[_buf_size],
                       num_bytes_2,
                       cudaMemcpyDeviceToDevice);
    }
    // advance the write pointer
    _write_index += bytes_written;
    if (_write_index >= _buf_size) {
        _write_index -= _buf_size;
    }

    // _buf_mutex.unlock();
}

void cuda_buffer::copy_items(std::shared_ptr<buffer> from, int nitems)
{
    std::lock_guard<std::mutex> guard(_buf_mutex);
    memcpy(write_ptr(), from->write_ptr(), nitems * _item_size);
}
} // namespace gr
