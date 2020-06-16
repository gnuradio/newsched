#include <gnuradio/blocklib/sync_block.hpp>

namespace gr
{
    sync_block::sync_block(const std::string& name)
        : block(name)
    {
        std::cout << "sync_block constructor" << std::endl;
    }
}