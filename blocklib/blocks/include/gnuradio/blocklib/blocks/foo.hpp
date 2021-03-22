#pragma once

#include <gnuradio/block_context.hpp>

namespace gr {
namespace blocks {
class foo : public block_context
{
    typedef std::shared_ptr<foo> sptr;
    public:
        static sptr make(int k);

};
} // namespace blocks
} // namespace gr