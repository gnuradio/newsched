#include <gnuradio/blocklib/blocks/fanout.hpp>

namespace gr {
namespace blocks {

template class fanout<std::int16_t>;
template class fanout<float>;
template class fanout<gr_complex>;

}
} // namespace gr