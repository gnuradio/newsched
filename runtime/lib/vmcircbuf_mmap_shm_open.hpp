#pragma once

#include <gnuradio/vmcircbuf.hpp>

namespace gr {
class vmcircbuf_mmap_shm_open : public vmcirc_buffer
{
private:
    int d_size;
    char* d_base;
public:
    typedef std::shared_ptr<vmcirc_buffer> sptr;
    vmcircbuf_mmap_shm_open(size_t num_items, size_t item_size, std::shared_ptr<buffer_properties> buf_properties);
    ~vmcircbuf_mmap_shm_open();
};

} // namespace gr
