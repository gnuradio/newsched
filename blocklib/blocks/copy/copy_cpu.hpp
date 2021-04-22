#pragma once

#include <gnuradio/blocks/copy.hpp>

namespace gr {
namespace blocks {

class copy_cpu : public copy
{
public:
    copy_cpu(size_t itemsize) : copy(itemsize) {}
    virtual work_return_code_t work(std::vector<block_work_input>& work_input,
                                    std::vector<block_work_output>& work_output) override;

protected:
    size_t d_itemsize;
};

// #define PARAM_LIST size_t itemsize
// #define PARAM_VALS itemsize


} // namespace blocks
} // namespace gr