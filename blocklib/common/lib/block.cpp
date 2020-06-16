
#include <gnuradio/blocklib/block.hpp>

namespace gr {
block::~block() {}
block::block(const std::string& name)
    : d_name(name)
{
}

} // namespace gr