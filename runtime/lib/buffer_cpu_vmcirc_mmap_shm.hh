#pragma once

#include <gnuradio/buffer_cpu_vmcirc.hh>

namespace gr {
class buffer_cpu_vmcirc_mmap_shm : public buffer_cpu_vmcirc
{
private:
    int d_size;
    char* d_base;
public:
    typedef std::shared_ptr<buffer_cpu_vmcirc> sptr;
    buffer_cpu_vmcirc_mmap_shm(size_t num_items, size_t item_size, std::shared_ptr<buffer_properties> buf_properties);
    ~buffer_cpu_vmcirc_mmap_shm();
};

} // namespace gr
