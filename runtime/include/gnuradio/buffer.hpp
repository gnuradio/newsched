#pragma once

#include <memory>

struct buffer_info_t {
    void* ptr;
    int n_items;
    size_t item_size; // is this necessary or stored in buffer
};

/**
 * @brief Abstract buffer class
 *
 */
class buffer
{
protected:
    std::string _name;

public:
    virtual void* read_ptr() = 0;
    virtual void* write_ptr() = 0;

    // virtual int capacity() = 0;
    // virtual int size() = 0;

    virtual buffer_info_t read_info() = 0;
    virtual buffer_info_t write_info() = 0;
    virtual void cancel() = 0;

    virtual void post_read(int num_items) = 0;
    virtual void post_write(int num_items) = 0;

    // This is not valid for all buffers, e.g. domain adapters
    virtual void copy_items(std::shared_ptr<buffer> from, int nitems) = 0;

    void set_name(const std::string& name) { _name = name; }
    std::string name() { return _name;}
};

typedef std::shared_ptr<buffer> buffer_sptr;