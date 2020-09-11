#pragma once

#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
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

    std::mutex _buf_mutex;
    std::vector<tag_t> _tags;

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
    }

    static buffer_sptr make(size_t num_items, size_t item_size, buffer_position_t buf_pos=buffer_position_t::NORMAL)
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
        // Need to lock the buffer to freeze the current state
        // if (!_buf_mutex.try_lock())
        // {
        //     return false;
        // }
        // _buf_mutex.lock();
        std::scoped_lock guard(_buf_mutex);

        info.ptr = read_ptr();
        info.n_items = size();
        info.item_size = _item_size;
        info.total_items = _total_read;

        return true;
    }

    virtual bool write_info(buffer_info_t& info)
    {
        // if (!_buf_mutex.try_lock())
        // {
        //     return false;
        // }
        // _buf_mutex.lock();
        std::scoped_lock guard(_buf_mutex);

        info.ptr = write_ptr();
        info.n_items = capacity() - size();
        info.item_size = _item_size;
        info.total_items = _total_written;

        return true;
    }

    virtual void cancel() {}
    // { _buf_mutex.unlock(); }

    virtual void post_read(int num_items)
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
        // _buf_mutex.unlock();
    }

    virtual void copy_items(std::shared_ptr<buffer> from, int nitems)
    {
        std::scoped_lock guard(_buf_mutex);

        memcpy(write_ptr(), from->write_ptr(), nitems * _item_size);
    }

    virtual std::vector<tag_t> get_tags(unsigned int num_items) override
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
    virtual void
    add_tags(unsigned int num_items,
             std::vector<tag_t>& tags) // overload with convenience functions later
        override
    {
        std::scoped_lock guard(_buf_mutex);

        for (auto tag : tags) {
            if (tag.offset < _total_written ||
                tag.offset >= _total_written + _num_items) {

            } else {
                _tags.push_back(tag);
            }
        }
    }
};

} // namespace gr
