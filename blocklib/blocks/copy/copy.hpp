#pragma once
#include <gnuradio/sync_block.hpp>
namespace gr {
namespace blocks {

class copy : public sync_block
{
public:
    typedef struct {
        size_t itemsize;
    } block_args;

    enum class available_impl { CPU, CUDA };

    typedef std::shared_ptr<copy> sptr;
    copy(block_args args) : sync_block("copy")
    {
        add_port(untyped_port::make("in", port_direction_t::INPUT, args.itemsize));
        add_port(untyped_port::make("out", port_direction_t::OUTPUT, args.itemsize));
    }

    static sptr make(block_args args, available_impl impl = available_impl::CPU)
    {
        switch (impl) {
        case available_impl::CPU:
            make_cpu(args);
            break;
        // case available_impl::CUDA;
        // make_cuda(args);
        // break;
        default:
            throw std::invalid_argument(
                "blocks::copy - invalid implementation specified");
        }
    }

    /**
     * @brief Set the implementation to CPU and return a shared pointer to the block
     * instance
     *
     * @return std::shared_ptr<copy>
     */
    static sptr make_cpu(block_args args);
};

} // namespace blocks
} // namespace gr
