
#include <gnuradio/blocklib/block.hpp>

namespace gr {
// block::~block() {}
block::block(const std::string& name) : node(name) {}

} // namespace gr