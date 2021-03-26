#include <gnuradio/blocks/copy.hpp>
#include <gnuradio/sync_block.hpp>
namespace gr {
namespace blocks {
namespace impl {
class copy_impl : public sync_block
{
    public:
    copy_impl(size_t itemsize);
};
} // namespace impl
} // namespace blocks
} // namespace gr