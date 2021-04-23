#pragma once
#include <gnuradio/sync_block.hpp>
namespace gr {
namespace blocks {

class copy : public sync_block
{
public:
    typedef std::shared_ptr<copy> sptr;
    copy(size_t itemsize) : sync_block("copy")
    {
        add_port(untyped_port::make("in", port_direction_t::INPUT, itemsize));
        add_port(untyped_port::make("out", port_direction_t::OUTPUT, itemsize));
    }

    /**
     * @brief Set the implementation to CPU and return a shared pointer to the block
     * instance
     *
     * @return std::shared_ptr<copy>
     */
    static sptr make_cpu(size_t itemsize);
};

} // namespace blocks
} // namespace gr
