#include <gnuradio/blocklib/blocks/dummy.hpp>

namespace gr {
namespace blocks {

template class dummy<std::int16_t>;
template class dummy<float>;
template class dummy<gr_complex>;

}
} // namespace gr