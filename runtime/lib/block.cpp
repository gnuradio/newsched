
#include <gnuradio/block.hpp>

namespace gr {
block::~block() {}
block::block(const std::string& name,
             const io_signature& input_signature,
             const io_signature& output_signature)
    : d_name(name),
      d_input_signature(input_signature),
      d_output_signature(output_signature)
{
}
} // namespace gr