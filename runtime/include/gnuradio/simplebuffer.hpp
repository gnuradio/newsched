#pragma once

#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <gnuradio/buffer.hpp>

namespace gr {

class simplebuffer_reader;
class simplebuffer : public buffer
{
private:
    std::vector<uint8_t> _buffer;

public:
    typedef std::shared_ptr<simplebuffer> sptr;
    simplebuffer(size_t num_items,
                 size_t item_size,
                 std::shared_ptr<buffer_properties> buf_properties)
        : buffer(num_items, item_size, buf_properties)
    {
        _buffer.resize(_buf_size * 2); // double circular buffer
        _write_index = 0;

        set_type("simplebuffer");
    }

    static buffer_sptr make(size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties)
    {
        return buffer_sptr(new simplebuffer(num_items, item_size, buffer_properties));
    }

    void* read_ptr(size_t index) { return (void*)&_buffer[index]; }
    void* write_ptr() { return (void*)&_buffer[_write_index]; }

    virtual void post_write(int num_items)
    {
        std::scoped_lock guard(_buf_mutex);

        size_t bytes_written = num_items * _item_size;
        size_t wi1 = _write_index;
        size_t wi2 = _write_index + _buf_size;
        // num_items were written to the buffer
        // copy the data to the second half of the buffer

        size_t num_bytes_1 = std::min(_buf_size - wi1, bytes_written);
        size_t num_bytes_2 = bytes_written - num_bytes_1;

        memcpy(&_buffer[wi2], &_buffer[wi1], num_bytes_1);
        if (num_bytes_2)
            memcpy(&_buffer[0], &_buffer[_buf_size], num_bytes_2);


        // advance the write pointer
        _write_index += bytes_written;
        if (_write_index >= _buf_size) {
            _write_index -= _buf_size;
        }

        _total_written += num_items;
    }

    virtual std::shared_ptr<buffer_reader>
    add_reader(std::shared_ptr<buffer_properties> buf_props);
};

class simplebuffer_reader : public buffer_reader
{
public:
    simplebuffer_reader(buffer_sptr buffer,
                        std::shared_ptr<buffer_properties> buf_props,
                        size_t read_index = 0)
        : buffer_reader(buffer, buf_props, read_index)
    {
    }

    virtual void post_read(int num_items)
    {
        std::scoped_lock guard(_rdr_mutex);

        // advance the read pointer
        _read_index += num_items * _buffer->item_size();
        if (_read_index >= _buffer->buf_size()) {
            _read_index -= _buffer->buf_size();
        }
        _total_read += num_items;
    }
};

} // namespace gr
