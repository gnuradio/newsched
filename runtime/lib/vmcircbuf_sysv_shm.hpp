#pragma once

#include <gnuradio/vmcircbuf.hpp>

namespace gr {
class vmcircbuf_sysv_shm : public vmcirc_buffer
{

public:
    typedef std::shared_ptr<vmcirc_buffer> sptr;
    vmcircbuf_sysv_shm(size_t num_items, size_t item_size);
    ~vmcircbuf_sysv_shm();
};

} // namespace gr
