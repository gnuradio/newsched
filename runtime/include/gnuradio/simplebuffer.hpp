#pragma once

#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <gnuradio/buffer.hpp>

namespace gr {
class simplebuffer : public buffer
{
private:
    std::vector<uint8_t> _buffer;
    unsigned int _read_index;
    unsigned int _write_index;
    unsigned int _num_items;
    unsigned int _item_size;
    unsigned int _buf_size;

public:
    typedef std::shared_ptr<simplebuffer> sptr;
    simplebuffer(){};
    simplebuffer(size_t num_items, size_t item_size)
    {
        _num_items = num_items;
        _item_size = item_size;
        _buf_size = _num_items * _item_size;
        _buffer.resize(_buf_size * 2); // double circular buffer
        _read_index = 0;
        _write_index = 0;

        set_type("simplebuffer");
    }

    static buffer_sptr make(size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties)
    {
        return buffer_sptr(new simplebuffer(num_items, item_size));
    }

    int size()
    { // in number of items
        int w = _write_index;
        int r = _read_index;

        if (w < r)
            w += _buf_size;
        return (w - r) / _item_size;
    }
    int capacity() { return _num_items; }

    void* read_ptr() { return (void*)&_buffer[_read_index]; }
    void* write_ptr() { return (void*)&_buffer[_write_index]; }

    virtual bool read_info(buffer_info_t& info)
    {
        std::scoped_lock guard(_buf_mutex);

        info.ptr = read_ptr();
        info.n_items = size();
        info.item_size = _item_size;
        info.total_items = _total_read;

        return true;
    }

    virtual bool write_info(buffer_info_t& info)
    {
        std::scoped_lock guard(_buf_mutex);

        info.ptr = write_ptr();
        info.n_items = capacity() - size() - 1; // always keep the write pointer 1 behind the read ptr
        if (info.n_items < 0)
            info.n_items = 0;
        info.item_size = _item_size;
        info.total_items = _total_written;

        return true;
    }

    virtual void post_read(int num_items)
    {
        std::scoped_lock guard(_buf_mutex);

        // advance the read pointer
        _read_index += num_items * _item_size;
        if (_read_index >= _buf_size) {
            _read_index -= _buf_size;
        }
        _total_read += num_items;
    }
    virtual void post_write(int num_items)
    {
        std::scoped_lock guard(_buf_mutex);

        unsigned int bytes_written = num_items * _item_size;
        int wi1 = _write_index;
        int wi2 = _write_index + _buf_size;
        // num_items were written to the buffer
        // copy the data to the second half of the buffer

        int num_bytes_1 = std::min(_buf_size - wi1, bytes_written);
        int num_bytes_2 = bytes_written - num_bytes_1;

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

    virtual void copy_items(std::shared_ptr<buffer> from, int nitems)
    {
        std::scoped_lock guard(_buf_mutex);

        memcpy(write_ptr(), from->write_ptr(), nitems * _item_size);
    }
};

} // namespace gr
