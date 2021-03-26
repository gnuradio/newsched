#include <gnuradio/blocks/copy.hpp>
namespace gr {
namespace blocks {

class copy_impl : public copy
{
public:
    copy_impl(size_t itemsize) : copy(itemsize), _itemsize(itemsize) {}

protected:
    size_t _itemsize;
};

} // namespace blocks
} // namespace gr