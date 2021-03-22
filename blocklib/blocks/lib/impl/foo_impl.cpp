#include <gnuradio/blocklib/blocks/foo.hpp>

#include "../cpu/foo_cpu.hpp"

namespace gr {
namespace blocks {
foo::sptr foo::make(int k) {
    block_impl = std::static_pointer_cast<block>(std::make_shared<foo_cpu>(k));
}

} // namespace blocks
} // namespace gr