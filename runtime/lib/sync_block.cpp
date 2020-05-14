#include <gnuradio/sync_block.hpp>

namespace gr
{
    sync_block::sync_block(const std::string& name,
                           const io_signature& input_signature,
                           const io_signature& output_signature)
        : block(name, input_signature, output_signature)
    {
        std::cout << "sync_block constructor" << std::endl;
    }
}