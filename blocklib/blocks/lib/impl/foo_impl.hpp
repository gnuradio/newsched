#pragma once

#include <gnuradio/blocklib/blocks/foo.hpp>
#include <gnuradio/sync_block.hpp>

namespace gr
{
    namespace blocks
    {
        class foo_impl : public sync_block
        {
            public:
                foo_impl(int k) : sync_block("foo"), _k(k)
                {

                }
            protected:
                int _k;


        };
    }
}