#ifndef INCLUDED_SYNC_BLOCK_HPP
#define INCLUDED_SYNC_BLOCK_HPP

#include "block.hpp"

namespace gr {
class sync_block : virtual public block
{
public:
    sync_block(const std::string& name,
                           io_signature input_signature,
                           io_signature output_signature)
        : gr::block(name, input_signature, output_signature)
    {
    }
    ~sync_block();
};
} // namespace gr
#endif