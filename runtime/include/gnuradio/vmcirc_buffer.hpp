#pragma once

#include <string.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <gnuradio/buffer.hpp>


// Doubly mapped circular buffer class
// For now, just do this as the sysv_shm flavor
// expand out with the preferences and the factories later

namespace gr {
class vmcirc_buffer : public buffer
{
private:
    uint8_t* _buffer;

    unsigned int _read_index;
    unsigned int _write_index;
    unsigned int _num_items;
    unsigned int _item_size;
    unsigned int _buf_size;

    std::mutex _buf_mutex;
    std::vector<tag_t> _tags;

public:
    typedef std::shared_ptr<vmcirc_buffer> sptr;
    vmcirc_buffer(){};
    vmcirc_buffer(size_t num_items, size_t item_size);

    ~vmcirc_buffer();

    static buffer_sptr make(size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties)
    {
        // Figure out which factory to use - for now, just use sysv_shm

        return buffer_sptr(new vmcirc_buffer(num_items, item_size));
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
        info.n_items = capacity() - size() -
                       1; // always keep the write pointer 1 behind the read ptr
        if (info.n_items < 0)
            info.n_items = 0;
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
