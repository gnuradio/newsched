#pragma once

#include <gnuradio/buffer.hpp>
#include <mutex>
#include <string>

// Doubly mapped circular buffer class
// For now, just do this as the sysv_shm flavor
// expand out with the preferences and the factories later

namespace gr {

extern std::mutex s_vm_mutex;

// forward declaration of derived classes
// class vmcircbuf_sysv_shm;

enum class vmcirc_buffer_type { AUTO, SYSV_SHM, MMAP_SHM, MMAP_TMPFILE };

class vmcirc_buffer_properties : public buffer_properties
{
public:
    // typedef sptr std::shared_ptr<buffer_properties>;
    vmcirc_buffer_properties(vmcirc_buffer_type buffer_type_ = vmcirc_buffer_type::AUTO)
        : buffer_properties(), _buffer_type(buffer_type_)
    {
    }
    vmcirc_buffer_type buffer_type() { return _buffer_type; }
    static std::shared_ptr<buffer_properties> make(vmcirc_buffer_type buffer_type_)
    {
        return std::dynamic_pointer_cast<buffer_properties>(
            std::make_shared<vmcirc_buffer_properties>(buffer_type_));
    }

private:
    vmcirc_buffer_type _buffer_type;
};


class vmcirc_buffer : public buffer
{
protected:
    uint8_t* _buffer;

    unsigned int _read_index;
    unsigned int _write_index;
    unsigned int _num_items;
    unsigned int _item_size;
    unsigned int _buf_size;



public:
    typedef std::shared_ptr<vmcirc_buffer> sptr;


    static buffer_sptr make(size_t num_items,
                            size_t item_size,
                            std::shared_ptr<buffer_properties> buffer_properties);

    vmcirc_buffer(size_t num_items, size_t item_size);

    // These methods are common to all the vmcircbufs

    int size();

    int capacity();

    void* read_ptr();
    void* write_ptr();

    virtual bool read_info(buffer_info_t& info);

    virtual bool write_info(buffer_info_t& info);

    virtual void post_read(int num_items);
    virtual void post_write(int num_items);

    virtual void copy_items(std::shared_ptr<buffer> from, int nitems);

};

} // namespace gr

#define VMCIRC_BUFFER_ARGS \
    vmcirc_buffer::make, vmcirc_buffer_properties::make(vmcirc_buffer_type::AUTO)
#define VMCIRC_BUFFER_SYSV_SHM_ARGS \
    vmcirc_buffer::make, vmcirc_buffer_properties::make(vmcirc_buffer_type::SYSV_SHM)
#define VMCIRC_BUFFER_MMAP_SHM_ARGS \
    vmcirc_buffer::make, vmcirc_buffer_properties::make(vmcirc_buffer_type::MMAP_SHM)
