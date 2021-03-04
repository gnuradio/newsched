#pragma once

#include <gnuradio/scheduler_message.hpp>

namespace gr {

struct neighbor_interface
{
    neighbor_interface() {}
    virtual ~neighbor_interface() {}
    virtual void push_message(scheduler_message_sptr msg) = 0;
};
typedef std::shared_ptr<neighbor_interface> neighbor_interface_sptr;

} // namespace gr
