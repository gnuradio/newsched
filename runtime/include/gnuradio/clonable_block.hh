#pragma once

#include <gnuradio/block.hh>

namespace gr {


/**
 * @brief clonable interface to block
 *
 */
class clonable_block 
{
public:
    virtual std::shared_ptr<block> clone() const = 0;

};
} // namespace gr
