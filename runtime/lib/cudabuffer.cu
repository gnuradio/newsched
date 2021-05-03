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
cuda_buffer::cuda_buffer(size_t num_items,
                         size_t item_size,
                         cuda_buffer_type type,
                         std::shared_ptr<buffer_properties> buf_properties)
    : gr::buffer(num_items, item_size, buf_properties), _type(type)
{
    // _host_buffer.resize(_buf_size * 2); // double circular buffer
    cudaMallocHost(
        &_host_buffer,
        _buf_size *
            2); // double circular buffer - should do something more intelligent here
    cudaMalloc(
        &_device_buffer,
        _buf_size *
            2); // double circular buffer - should do something more intelligent here
    set_type("cuda_buffer_" + std::to_string((int)_type));

    cudaStreamCreate(&stream);
}
cuda_buffer::~cuda_buffer()
{
    cudaFree(_device_buffer);
    cudaFree(_host_buffer);
}

buffer_sptr cuda_buffer::make(size_t num_items,
                              size_t item_size,
                              std::shared_ptr<buffer_properties> buffer_properties)
{
    auto cbp = std::static_pointer_cast<cuda_buffer_properties>(buffer_properties);
    if (cbp != nullptr) {
        return buffer_sptr(
            new cuda_buffer(num_items, item_size, cbp->buffer_type(), buffer_properties));
    } else {
        throw std::runtime_error(
            "Failed to cast buffer properties to cuda_buffer_properties");
    }
}

void* cuda_buffer::read_ptr(size_t index)
{
    if (_type == cuda_buffer_type::D2H) {
        return (void*)&_host_buffer[index];
    } else {
        return (void*)&_device_buffer[index];
    }
}
void* cuda_buffer::write_ptr()
{
    if (_type == cuda_buffer_type::H2D) {
        return (void*)&_host_buffer[_write_index];
    } else {
        return (void*)&_device_buffer[_write_index];
    }
}

void cuda_buffer_reader::post_read(int num_items)
{
    std::lock_guard<std::mutex> guard(_rdr_mutex);
    // advance the read pointer
    _read_index += num_items * _buffer->item_size();
    if (_read_index >= _buffer->buf_size()) {
        _read_index -= _buffer->buf_size();
    }
    _total_read += num_items;
}
void cuda_buffer::post_write(int num_items)
{
    std::lock_guard<std::mutex> guard(_buf_mutex);

    size_t bytes_written = num_items * _item_size;
    size_t wi1 = _write_index;
    size_t wi2 = _write_index + _buf_size;
    // num_items were written to the buffer
    // copy the data to the second half of the buffer

    size_t num_bytes_1 = std::min(_buf_size - wi1, bytes_written);
    size_t num_bytes_2 = bytes_written - num_bytes_1;

    if (_type == cuda_buffer_type::H2D) {
        cudaMemcpyAsync(&_device_buffer[wi1],
                        &_host_buffer[wi1],
                        bytes_written,
                        cudaMemcpyHostToDevice,
                        stream);

        // memcpy(&_host_buffer[wi2], &_host_buffer[wi1], num_bytes_1);
        cudaMemcpyAsync(&_device_buffer[wi2],
                        &_device_buffer[wi1],
                        num_bytes_1,
                        cudaMemcpyDeviceToDevice,
                        stream);
        if (num_bytes_2) {
            // memcpy(&_host_buffer[0], &_host_buffer[_buf_size], num_bytes_2);
            cudaMemcpyAsync(&_device_buffer[0],
                            &_device_buffer[_buf_size],
                            num_bytes_2,
                            cudaMemcpyDeviceToDevice,
                            stream);
        }
    } else if (_type == cuda_buffer_type::D2H) {
        cudaMemcpyAsync(&_host_buffer[wi1],
                        &_device_buffer[wi1],
                        bytes_written,
                        cudaMemcpyDeviceToHost,
                        stream);

        memcpy(&_host_buffer[wi2], &_host_buffer[wi1], num_bytes_1);

        if (num_bytes_2) {
            memcpy(&_host_buffer[0], &_host_buffer[_buf_size], num_bytes_2);
        }
    } else // D2D
    {
        cudaMemcpyAsync(&_device_buffer[wi2],
                        &_device_buffer[wi1],
                        num_bytes_1,
                        cudaMemcpyDeviceToDevice);
        if (num_bytes_2)
            cudaMemcpyAsync(&_device_buffer[0],
                            &_device_buffer[_buf_size],
                            num_bytes_2,
                            cudaMemcpyDeviceToDevice,
                            stream);
    }
    // advance the write pointer
    _write_index += bytes_written;
    if (_write_index >= _buf_size) {
        _write_index -= _buf_size;
    }
    cudaStreamSynchronize(stream);
}

std::shared_ptr<buffer_reader>
cuda_buffer::add_reader(std::shared_ptr<buffer_properties> buf_props)
{
    std::shared_ptr<cuda_buffer_reader> r(
        new cuda_buffer_reader(shared_from_this(), buf_props, _write_index));
    _readers.push_back(r.get());
    return r;
}

} // namespace gr
